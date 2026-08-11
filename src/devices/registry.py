"""Catalogue of supported headsets and what each one can actually decode.

Every entry records the electrode layout the hardware ships with, so that
:func:`describe_device` can tell you *before you buy* whether the device can
see left-vs-right hand motor imagery at all. Channel names and sampling rates
are read live from BrainFlow where possible rather than hard-coded, so this
stays correct as BrainFlow adds boards.
"""

from __future__ import annotations

from dataclasses import dataclass

from .channels import assess_montage

@dataclass(frozen=True)
class DeviceProfile:
    """What we know about a headset without plugging it in."""

    key: str
    display_name: str
    board_id_name: str
    default_channels: tuple[str, ...]
    sfreq: float
    electrodes_repositionable: bool
    approx_price_usd: str
    connection: str
    notes: str

    @property
    def montage_rating(self) -> str:
        return assess_montage(list(self.default_channels)).rating


#: Layouts below are the manufacturers' stock configurations. For boards whose
#: electrodes you place yourself (OpenBCI), the "default" is the layout their
#: own documentation ships with, and ``electrodes_repositionable`` records that
#: you are free — and for this project, strongly advised — to move them.
DEVICE_PROFILES: dict[str, DeviceProfile] = {
    "synthetic": DeviceProfile(
        key="synthetic",
        display_name="BrainFlow synthetic board (no hardware)",
        board_id_name="SYNTHETIC_BOARD",
        default_channels=(
            "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8",
            "F5", "F7", "F3", "F1", "F2", "F4", "F6", "F8",
        ),
        sfreq=250.0,
        electrodes_repositionable=False,
        approx_price_usd="free",
        connection="none",
        notes=(
            "Generates band-limited noise, not real motor imagery. Use it to "
            "exercise the acquisition and live-decoding plumbing end to end "
            "before hardware arrives. Accuracy on synthetic data is chance, "
            "and that is the correct result."
        ),
    ),
    "crown": DeviceProfile(
        key="crown",
        display_name="Neurosity Crown",
        board_id_name="CROWN_BOARD",
        default_channels=("CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"),
        sfreq=256.0,
        electrodes_repositionable=False,
        approx_price_usd="~$1,499",
        connection="Wi-Fi (device runs its own OS) or Bluetooth",
        notes=(
            "The only true headband-form-factor consumer device with C3/C4 AND "
            "CP3/CP4 over the sensorimotor strip — it is laid out for exactly "
            "this task. Dry comb electrodes reach through hair. Fastest path "
            "from unboxing to a working left/right decoder."
        ),
    ),
    "unicorn": DeviceProfile(
        key="unicorn",
        display_name="g.tec Unicorn Hybrid Black",
        board_id_name="UNICORN_BOARD",
        default_channels=("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"),
        sfreq=250.0,
        electrodes_repositionable=False,
        approx_price_usd="~$1,000-1,300 (quote from g.tec; EU pricing in EUR)",
        connection="Bluetooth",
        notes=(
            "Textbook BCI montage: Fz/C3/Cz/C4/Pz straddles the motor strip "
            "with a midline reference chain. Built by a BCI research company, "
            "well documented, widely used in published motor-imagery work. "
            "More 'headset' than 'headband' in form factor."
        ),
    ),
    "cyton": DeviceProfile(
        key="cyton",
        display_name="OpenBCI Cyton (8ch)",
        board_id_name="CYTON_BOARD",
        default_channels=("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"),
        sfreq=250.0,
        electrodes_repositionable=True,
        approx_price_usd="~$1,249 board + $350-500 headwear",
        connection="USB dongle (RFDuino radio)",
        notes=(
            "You choose every electrode position, so you can put all 8 over the "
            "sensorimotor strip (e.g. FC3/FC4, C3/C1/C2/C4, CP3/CP4) which no "
            "fixed-layout consumer device allows. Highest ceiling and the most "
            "educational, at the cost of setup time and gel or comb electrodes."
        ),
    ),
    "cyton_daisy": DeviceProfile(
        key="cyton_daisy",
        display_name="OpenBCI Cyton + Daisy (16ch)",
        board_id_name="CYTON_DAISY_BOARD",
        default_channels=(
            "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
            "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4",
        ),
        sfreq=125.0,
        electrodes_repositionable=True,
        approx_price_usd="~$2,499 board + headwear",
        connection="USB dongle",
        notes=(
            "16 channels lets CSP find genuinely good spatial filters. Note the "
            "sampling rate halves to 125 Hz when the Daisy is attached — still "
            "fine for the 8-30 Hz band this project uses."
        ),
    ),
    "ganglion": DeviceProfile(
        key="ganglion",
        display_name="OpenBCI Ganglion (4ch)",
        board_id_name="GANGLION_BOARD",
        default_channels=("C3", "C4", "CP3", "CP4"),
        sfreq=200.0,
        electrodes_repositionable=True,
        approx_price_usd="~$715 all-in ($625 board + $20 dongle + $70 electrodes/paste)",
        connection="Bluetooth LE dongle (sold separately, mandatory)",
        notes=(
            "Cheapest route to electrodes you can place at C3/C4 yourself. Four "
            "channels is the practical minimum: CSP has almost nothing to work "
            "with, so expect band-power and simple Laplacian features to do as "
            "well as anything fancier. The board price includes neither the dongle "
            "nor any electrodes. The layout listed here is the motor-imagery "
            "placement to wire up, not a factory default — see "
            "docs/GANGLION_QUICKSTART.md."
        ),
    ),
    "pieeg": DeviceProfile(
        key="pieeg",
        display_name="PiEEG (Raspberry Pi shield, 8ch)",
        board_id_name="PIEEG_BOARD",
        default_channels=("C3", "C4", "CP3", "CP4", "FC3", "FC4", "C5", "C6"),
        sfreq=250.0,
        electrodes_repositionable=True,
        approx_price_usd="~$350 board + ~$60 Raspberry Pi + electrodes",
        connection="Raspberry Pi GPIO (the Pi runs the software)",
        notes=(
            "Cheapest route to a real ADS1299 amplifier — the same converter chip as "
            "OpenBCI's Cyton, at a third of the price. You supply the Pi, the "
            "electrodes and the headwear, and you place every electrode yourself, so "
            "the listed layout is the motor-imagery placement to aim for, not a "
            "factory default. CAVEAT: the 8-channel board is discontinued and stock "
            "is intermittent — check availability before planning around it. PiEEG-16 "
            "(~$390) is the current product."
        ),
    ),
    "diy": DeviceProfile(
        key="diy",
        display_name="Home-built rig (import recordings from any source)",
        board_id_name="SYNTHETIC_BOARD",
        default_channels=(),
        sfreq=250.0,
        electrodes_repositionable=True,
        approx_price_usd="~$40-150 depending on channel count",
        connection="whatever you built — import files rather than streaming live",
        notes=(
            "For rigs this project cannot drive directly (BioAmp EXG Pill on an "
            "Arduino/ESP32, a bare ADS1299 breakout, anything home-etched). Record "
            "with your own firmware, export CSV, and bring it in with "
            "`cli import` plus a cue-times file. Everything downstream then works "
            "identically. Two well-placed channels at C3 and C4 are enough to try "
            "this task; the hard parts are amplifier noise and electrode contact, "
            "not channel count."
        ),
    ),
    "ganglion_native": DeviceProfile(
        key="ganglion_native",
        display_name="OpenBCI Ganglion (4ch, no dongle — native Bluetooth)",
        board_id_name="GANGLION_NATIVE_BOARD",
        default_channels=("C3", "C4", "CP3", "CP4"),
        sfreq=200.0,
        electrodes_repositionable=True,
        approx_price_usd="same board, saves the ~$20 dongle",
        connection="your computer's built-in Bluetooth (needs --mac-address)",
        notes=(
            "Identical hardware to `ganglion`, connected through your laptop's own "
            "Bluetooth instead of OpenBCI's USB dongle. Try this first if the dongle "
            "is out of stock or you would rather not buy one; fall back to `ganglion` "
            "if the connection proves flaky, which it can be on some Bluetooth stacks. "
            "Needs the board's MAC address, printed on the board and discoverable "
            "with any BLE scanner app."
        ),
    ),
    "muse2": DeviceProfile(
        key="muse2",
        display_name="Muse 2",
        board_id_name="MUSE_2_BOARD",
        default_channels=("TP9", "AF7", "AF8", "TP10"),
        sfreq=256.0,
        electrodes_repositionable=False,
        approx_price_usd="~$250",
        connection="Bluetooth LE",
        notes=(
            "CANNOT do this task. Two forehead and two behind-the-ear electrodes, "
            "nothing within ~7 cm of the hand area of motor cortex. Excellent for "
            "alpha, blinks, jaw clench, drowsiness and meditation; the wrong tool "
            "for left-vs-right hand imagery. If a model appears to work on Muse "
            "data, it has learned your eye or jaw movements."
        ),
    ),
    "muse_s": DeviceProfile(
        key="muse_s",
        display_name="Muse S",
        board_id_name="MUSE_S_BOARD",
        default_channels=("TP9", "AF7", "AF8", "TP10"),
        sfreq=256.0,
        electrodes_repositionable=False,
        approx_price_usd="~$400",
        connection="Bluetooth LE",
        notes="Same electrode layout as Muse 2, same limitation. Softer band for sleep use.",
    ),
    "freeeeg32": DeviceProfile(
        key="freeeeg32",
        display_name="FreeEEG32 (open hardware, 32ch)",
        board_id_name="FREEEEG32_BOARD",
        default_channels=(),
        sfreq=512.0,
        electrodes_repositionable=True,
        approx_price_usd="~$500-800 depending on source",
        connection="USB",
        notes=(
            "32 channels for the price of an 8-channel commercial board, but you "
            "supply the cap, electrodes and patience. A good option only if "
            "somebody in the group has soldered and debugged hardware before."
        ),
    ),
}


def get_profile(key: str) -> DeviceProfile:
    try:
        return DEVICE_PROFILES[key]
    except KeyError:
        raise KeyError(
            f"Unknown device '{key}'. Known devices: {', '.join(sorted(DEVICE_PROFILES))}"
        ) from None


def resolve_board_id(key: str) -> int:
    """Look up the BrainFlow integer board id for a device key."""
    from brainflow.board_shim import BoardIds  # imported lazily; see module docstring

    profile = get_profile(key)
    return int(getattr(BoardIds, profile.board_id_name).value)


def describe_device(key: str) -> str:
    """Human-readable buying/suitability report for one device."""
    p = get_profile(key)
    lines = [
        f"{p.display_name}  [{p.key}]",
        f"  price       : {p.approx_price_usd}",
        f"  connection  : {p.connection}",
        f"  sample rate : {p.sfreq:g} Hz",
        "  electrodes  : "
        + (", ".join(p.default_channels) if p.default_channels else "(user supplied)")
        + ("  (repositionable)" if p.electrodes_repositionable else "  (fixed)"),
        "",
    ]
    if p.default_channels:
        lines.append(assess_montage(list(p.default_channels)).describe())
        lines.append("")
    lines.append(f"  {p.notes}")
    return "\n".join(lines)


def describe_all() -> str:
    """Suitability report for every catalogued device, best montage first."""
    order = {"good": 0, "marginal": 1, "unsuitable": 2}
    keys = sorted(
        DEVICE_PROFILES,
        key=lambda k: (order.get(DEVICE_PROFILES[k].montage_rating, 3), k),
    )
    return "\n\n".join(describe_device(k) for k in keys)
