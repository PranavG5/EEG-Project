/*
 * NeuroDecode frontend.
 *
 * Deliberately dependency-free and modular so the app stays easy to extend:
 *
 *   api      — every backend call lives here. New endpoint -> add one method.
 *   state    — the single source of truth for the UI.
 *   render.* — one small function per results panel. New visualisation ->
 *              write a render function and call it from renderResults().
 *
 * The decoder list and all result panels are built from whatever the API
 * returns, so adding a decoder (or a field) on the backend surfaces here with
 * no frontend changes.
 */

// ── API layer ────────────────────────────────────────────────────────────
const api = {
  async health() {
    const r = await fetch('/api/health');
    if (!r.ok) throw new Error('API unhealthy');
    return r.json();
  },
  async decoders() {
    const r = await fetch('/api/decoders');
    if (!r.ok) throw new Error('Could not load decoders');
    return r.json();
  },
  async subjects() {
    const r = await fetch('/api/subjects');
    if (!r.ok) throw new Error('Could not load subjects');
    return r.json();
  },
  async decode({ decoder, subject, files }) {
    const fd = new FormData();
    fd.append('decoder', decoder);
    if (files && files.length) {
      for (const f of files) fd.append('files', f, f.name);
    } else {
      fd.append('subject', String(subject));
    }
    const r = await fetch('/api/decode', { method: 'POST', body: fd });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || `Request failed (${r.status})`);
    return body;
  },
};

// ── State ────────────────────────────────────────────────────────────────
const state = {
  source: 'demo',      // 'demo' | 'upload'
  subject: 7,
  decoder: 'csp_lda',
  files: [],
  busy: false,
};

const $ = (sel) => document.querySelector(sel);
const fmtPct = (x) => `${(x * 100).toFixed(1)}%`;
const handClass = (name) => (name || '').startsWith('left') ? 'left' : 'right';
const handLabel = (name) =>
  (name || '').startsWith('left') ? 'LEFT' : 'RIGHT';

// ── View switching ───────────────────────────────────────────────────────
function showView(which) {
  for (const id of ['empty-state', 'loading-state', 'error-state', 'results']) {
    $(`#${id}`).classList.toggle('hidden', id !== which);
  }
}

// ── Renderers (one per panel — add new ones here) ────────────────────────
const render = {
  stats(res) {
    const delta = res.accuracy - res.chance;
    const cells = [
      ['Source', res.source],
      ['Trials', res.n_trials],
      ['Channels', res.n_channels],
      ['Sampling rate', `${res.sfreq.toFixed(0)} Hz`],
      ['Chance level', fmtPct(res.chance)],
      ['Above chance', `${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)} pts`],
    ];
    $('#stat-row').innerHTML = cells.map(([lab, val]) => `
      <div class="stat"><div class="s-lab">${lab}</div>
        <div class="s-val">${val}</div></div>`).join('');
  },

  accuracyBadge(res) {
    const good = res.accuracy >= res.chance + 0.05;
    $('#accuracy-badge').innerHTML = `
      <div class="acc-val" style="color:${good ? 'var(--good)' : 'var(--text)'}">
        ${fmtPct(res.accuracy)}</div>
      <div class="acc-lab">held-out accuracy</div>`;
  },

  trials(res) {
    $('#trial-body').innerHTML = res.trials.map((t) => `
      <tr class="${t.correct ? '' : 'wrong'}">
        <td class="muted">${t.index}</td>
        <td><span class="hand ${handClass(t.imagined)}">${handLabel(t.imagined)}</span></td>
        <td><span class="hand ${handClass(t.decoded)}">${handLabel(t.decoded)}</span></td>
        <td>
          <div class="conf-cell">
            <div class="conf-track">
              <div class="conf-fill" style="width:${(t.confidence * 100).toFixed(0)}%"></div>
            </div>
            <span class="conf-num">${(t.confidence * 100).toFixed(0)}%</span>
          </div>
        </td>
        <td class="tick">${t.correct ? '✅' : '❌'}</td>
      </tr>`).join('');
  },

  confusion(res) {
    const cm = res.confusion_matrix;
    const names = res.class_names.map(handLabel);
    const max = Math.max(...cm.flat(), 1);
    const cell = (v) => {
      const a = 0.12 + 0.78 * (v / max);
      return `<div class="cm-cell" style="background:rgba(91,140,255,${a.toFixed(3)})">${v}</div>`;
    };
    $('#confusion').innerHTML = `
      <div></div>
      <div class="cm-label">pred ${names[0]}</div>
      <div class="cm-label">pred ${names[1]}</div>
      <div class="cm-label row">true ${names[0]}</div>${cell(cm[0][0])}${cell(cm[0][1])}
      <div class="cm-label row">true ${names[1]}</div>${cell(cm[1][0])}${cell(cm[1][1])}`;
  },

  classBars(res) {
    const cm = res.confusion_matrix;
    const rows = res.class_names.map((name, i) => {
      const total = cm[i].reduce((a, b) => a + b, 0) || 1;
      const recall = cm[i][i] / total;
      const color = handClass(name) === 'left' ? 'var(--left)' : 'var(--right)';
      return `<div class="cb-row">
        <span class="hand ${handClass(name)}">${handLabel(name)}</span>
        <div class="cb-track">
          <div class="cb-fill" style="width:${(recall * 100).toFixed(0)}%;background:${color}"></div>
        </div>
        <span class="cb-num">${(recall * 100).toFixed(0)}%</span>
      </div>`;
    });
    $('#class-bars').innerHTML = rows.join('');
  },

  topomap(res) {
    const img = $('#topomap');
    if (res.topomap_png) {
      img.src = res.topomap_png;
      img.classList.remove('hidden');
    } else {
      img.classList.add('hidden');
    }
  },
};

function renderResults(res) {
  $('#result-title').textContent = `Results — ${res.decoder_name}`;
  $('#result-sub').textContent =
    `${res.source} · ${res.n_trials} trials · 5-fold cross-validated`;
  render.accuracyBadge(res);
  render.stats(res);
  render.trials(res);
  render.confusion(res);
  render.classBars(res);
  render.topomap(res);
  showView('results');
}

// ── Controls ─────────────────────────────────────────────────────────────
function setSource(source) {
  state.source = source;
  document.querySelectorAll('.seg-btn').forEach((b) =>
    b.classList.toggle('active', b.dataset.source === source));
  $('#source-demo').classList.toggle('hidden', source !== 'demo');
  $('#source-upload').classList.toggle('hidden', source !== 'upload');
  updateRunButton();
}

function updateRunButton() {
  const ready = state.source === 'demo' || state.files.length > 0;
  $('#run-btn').disabled = state.busy || !ready;
  $('#status-line').textContent = state.busy
    ? ''
    : (ready ? '' : 'Add at least one .edf file to decode.');
}

function renderFileList() {
  $('#file-list').innerHTML = state.files.map((f) => `
    <li><span>📄</span><span>${f.name}</span>
      <span class="fsize">${(f.size / 1024 / 1024).toFixed(1)} MB</span></li>`).join('');
}

function addFiles(fileList) {
  const edfs = Array.from(fileList).filter((f) => f.name.toLowerCase().endsWith('.edf'));
  state.files = state.files.concat(edfs);
  renderFileList();
  updateRunButton();
}

async function runDecode() {
  state.busy = true;
  updateRunButton();
  const isNet = state.decoder === 'eegnet';
  $('#loading-title').textContent = isNet
    ? 'Training EEGNet…' : 'Calibrating decoder…';
  $('#loading-sub').textContent = isNet
    ? 'Training a CNN on each cross-validation fold — this takes a minute.'
    : 'Filtering signal, extracting features, cross-validating.';
  showView('loading-state');

  try {
    const res = await api.decode({
      decoder: state.decoder,
      subject: state.subject,
      files: state.source === 'upload' ? state.files : null,
    });
    renderResults(res);
  } catch (err) {
    $('#error-message').textContent = err.message || String(err);
    showView('error-state');
  } finally {
    state.busy = false;
    updateRunButton();
  }
}

// ── Init ─────────────────────────────────────────────────────────────────
async function init() {
  // Health indicator
  api.health()
    .then(() => $('#health-dot').classList.add('ok'))
    .catch(() => $('#health-dot').classList.add('down'));

  // Decoders -> selectable cards (driven entirely by the API)
  try {
    const decoders = await api.decoders();
    $('#decoder-list').innerHTML = decoders.map((d) => `
      <button class="decoder-card${d.key === state.decoder ? ' selected' : ''}"
              data-key="${d.key}">
        <div class="dc-name">${d.name}</div>
        <div class="dc-desc">${d.description}</div>
      </button>`).join('');
    document.querySelectorAll('.decoder-card').forEach((card) => {
      card.addEventListener('click', () => {
        state.decoder = card.dataset.key;
        document.querySelectorAll('.decoder-card').forEach((c) =>
          c.classList.toggle('selected', c === card));
      });
    });
  } catch (err) {
    $('#decoder-list').innerHTML =
      `<p class="hint">Could not reach the API: ${err.message}</p>`;
  }

  // Demo subjects
  try {
    const { subjects } = await api.subjects();
    $('#subject').innerHTML = subjects.map((s) =>
      `<option value="${s}"${s === state.subject ? ' selected' : ''}>Subject ${s}</option>`
    ).join('');
    $('#subject').addEventListener('change', (e) => {
      state.subject = Number(e.target.value);
    });
  } catch { /* health dot already reflects this */ }

  // Source toggle
  document.querySelectorAll('.seg-btn').forEach((b) =>
    b.addEventListener('click', () => setSource(b.dataset.source)));

  // Upload interactions
  const dz = $('#dropzone');
  const input = $('#file-input');
  dz.addEventListener('click', () => input.click());
  input.addEventListener('change', (e) => addFiles(e.target.files));
  ['dragenter', 'dragover'].forEach((ev) =>
    dz.addEventListener(ev, (e) => {
      e.preventDefault(); dz.classList.add('dragover');
    }));
  ['dragleave', 'drop'].forEach((ev) =>
    dz.addEventListener(ev, (e) => {
      e.preventDefault(); dz.classList.remove('dragover');
    }));
  dz.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));

  $('#run-btn').addEventListener('click', runDecode);
  updateRunButton();
}

init();
