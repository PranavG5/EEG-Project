"""BrainFlow-backed :class:`EEGSource` implementation.

BrainFlow is the reason this project does not care which headband gets bought.
It exposes OpenBCI, Neurosity, g.tec Unicorn, Muse, BrainBit, Mentalab and a
few dozen others behind one API, so the choice of hardware becomes a
command-line flag rather than a rewrite. (Notably it does *not* support EMOTIV,
whose SDK is proprietary — one more reason EMOTIV is not recommended here.)
"""

from __future__ import annotations

import time

import numpy as np

from .base import EEGSource, StreamInfo, eeg_channel_subset
from .registry import get_profile, resolve_board_id


class BrainFlowSource(EEGSource):
    """Live EEG from any BrainFlow-supported board.

    Parameters
    ----------
    device_key
        Key into :data:`~src.devices.registry.DEVICE_PROFILES`, e.g. ``"crown"``.
    serial_port
        Serial device for dongle-attached boards (OpenBCI Cyton/Ganglion), e.g.
        ``"/dev/ttyUSB0"`` on Linux, ``"COM3"`` on Windows, ``"/dev/cu.usbserial-*"``
        on macOS. Ignored by Bluetooth/Wi-Fi boards.
    mac_address, ip_address, serial_number
        Board-specific addressing; see BrainFlow's docs per board. Muse needs a
        MAC on some platforms, Neurosity needs the device name/credentials via
        environment variables.
    channel_override
        Replace the board's declared electrode names. Essential for OpenBCI,
        where the software has no idea where you physically stuck the
        electrodes — if you moved them to the motor strip, say so here or every
        montage check and channel mapping downstream will be wrong.
    """

    def __init__(
        self,
        device_key: str,
        *,
        serial_port: str = "",
        mac_address: str = "",
        ip_address: str = "",
        ip_port: int = 0,
        serial_number: str = "",
        timeout: int = 15,
        channel_override: list[str] | None = None,
    ) -> None:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams

        self._profile = get_profile(device_key)
        self._board_id = resolve_board_id(device_key)

        params = BrainFlowInputParams()
        params.serial_port = serial_port
        params.mac_address = mac_address
        params.ip_address = ip_address
        params.ip_port = ip_port
        params.serial_number = serial_number
        params.timeout = timeout

        self._board = BoardShim(self._board_id, params)
        self._sfreq = float(BoardShim.get_sampling_rate(self._board_id))
        self._eeg_rows = list(BoardShim.get_eeg_channels(self._board_id))

        raw_names = self._board_channel_names(BoardShim)
        if channel_override is not None:
            if len(channel_override) != len(self._eeg_rows):
                raise ValueError(
                    f"channel_override has {len(channel_override)} names but "
                    f"{self._profile.display_name} exposes {len(self._eeg_rows)} EEG channels"
                )
            raw_names = list(channel_override)

        keep, canonical = eeg_channel_subset(raw_names)
        if not canonical:
            raise ValueError(
                f"None of {raw_names} were recognised as 10-05 electrode names. "
                "Pass channel_override with the positions you actually used."
            )
        # Map back onto BrainFlow's row indices in the returned data matrix.
        self._rows = [self._eeg_rows[i] for i in keep]
        self._info = StreamInfo(
            name=self._profile.display_name,
            sfreq=self._sfreq,
            ch_names=canonical,
        )
        self._started = False

    def _board_channel_names(self, board_shim_cls: type) -> list[str]:
        """Electrode names the board declares, falling back to the profile."""
        try:
            names = board_shim_cls.get_eeg_names(self._board_id)
            if isinstance(names, str):
                names = names.split(",")
            if names:
                return [n.strip() for n in names]
        except Exception:
            pass
        if self._profile.default_channels:
            return list(self._profile.default_channels)
        raise ValueError(
            f"{self._profile.display_name} does not declare electrode names and no "
            "defaults are known. Pass channel_override explicitly."
        )

    # -- EEGSource contract ----------------------------------------------

    @property
    def info(self) -> StreamInfo:
        return self._info

    def start(self) -> None:
        if self._started:
            return
        self._board.prepare_session()
        self._board.start_stream()
        self._started = True
        # Discard the first moments: amplifiers settle, filters ring, and
        # Bluetooth stacks deliver a burst of buffered junk on connect.
        time.sleep(1.0)
        self._board.get_board_data()

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self._board.stop_stream()
        finally:
            self._board.release_session()
            self._started = False

    def poll(self) -> np.ndarray:
        if not self._started:
            raise RuntimeError("poll() called before start()")
        data = self._board.get_board_data()  # drains BrainFlow's internal buffer
        if data.size == 0:
            return np.zeros((len(self._rows), 0))
        # BrainFlow reports EEG in microvolts; the rest of this codebase and all
        # of MNE work in volts.
        return data[self._rows, :] * 1e-6


class ArraySource(EEGSource):
    """Replay an in-memory array as if it were a live device.

    Used by the test suite and by ``--device synthetic`` demos so that the
    acquisition and live-decoding code paths can be exercised deterministically,
    without hardware and without BrainFlow's synthetic board's randomness.
    """

    def __init__(
        self,
        data: np.ndarray,
        sfreq: float,
        ch_names: list[str],
        *,
        name: str = "array replay",
        realtime: bool = False,
    ) -> None:
        if data.ndim != 2 or data.shape[0] != len(ch_names):
            raise ValueError(
                f"data must be (n_channels, n_samples) matching {len(ch_names)} names, "
                f"got {data.shape}"
            )
        self._data = data
        self._pos = 0
        self._realtime = realtime
        self._last_poll: float | None = None
        self._info = StreamInfo(name=name, sfreq=sfreq, ch_names=list(ch_names))

    @property
    def info(self) -> StreamInfo:
        return self._info

    def start(self) -> None:
        self._pos = 0
        self._last_poll = time.monotonic()

    def stop(self) -> None:
        self._last_poll = None

    def poll(self) -> np.ndarray:
        if self._pos >= self._data.shape[1]:
            return np.zeros((self._data.shape[0], 0))
        if self._realtime:
            now = time.monotonic()
            elapsed = now - (self._last_poll or now)
            self._last_poll = now
            n = int(elapsed * self._info.sfreq)
        else:
            n = int(0.25 * self._info.sfreq)
        n = max(n, 0)
        chunk = self._data[:, self._pos : self._pos + n]
        self._pos += chunk.shape[1]
        return chunk
