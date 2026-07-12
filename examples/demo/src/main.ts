/**
 * Demo orchestration: wires the panels to pca-web's public API — Worker
 * proxies ('pca-web/client'), main-thread sync/async fits ('pca-web'),
 * progress snapshots, aborts, IncrementalPCA streaming, and model
 * import/export/IndexedDB persistence.
 */
import type {
  AnyPCAModel,
  FloatArray,
  IncrementalPCAModel,
  IncrementalPCAOptions,
  MatrixInput,
  PCAFitProgress,
  PCAModel,
  PCAOptions,
  SvdSolver,
} from 'pca-web';
import { IncrementalPCA, Matrix, modelFromJSON, modelToJSON, PCA } from 'pca-web';
import type { WorkerFitOptions } from 'pca-web/client';
import { WorkerIncrementalPCA, WorkerPCA } from 'pca-web/client';
import {
  createHistogram,
  createMseLine,
  createScatter,
  createScree,
  renderTiles,
  resizeAllCharts,
} from './charts';
import {
  type Dataset,
  type DemoDtype,
  isPresetKind,
  loadDigits,
  makeSynthetic,
  PRESET_INFO,
} from './data';
import {
  addRunRow,
  bindSliderOutput,
  byId,
  download,
  fmtMs,
  inputEl,
  onRadioChange,
  readRadio,
  setProgress,
  setRadioDisabled,
  setStatus,
  startLiveness,
} from './ui';

// ---------------------------------------------------------------------
// Charts and static UI wiring
// ---------------------------------------------------------------------

const scatter = createScatter(byId('scatter'));
const scree = createScree(byId('scree'));
const mseLine = createMseLine(byId('mse-chart'));
const hist = createHistogram(byId('hist'));

startLiveness();
bindSliderOutput('synth-n', 'synth-n-out');
bindSliderOutput('synth-p', 'synth-p-out');
bindSliderOutput('synth-clusters', 'synth-clusters-out');
bindSliderOutput('recon-k', 'recon-k-out');

// ---------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------

interface RunConfig {
  estimator: 'pca' | 'ipca';
  mode: 'worker' | 'async' | 'sync';
  backend: 'cpu' | 'webgpu';
  nComponents: number | 'mle' | null;
  whiten: boolean;
  solver: SvdSolver;
  iteratedPower: number | 'auto';
  seed: number | null;
  batchSize: number;
}

function parseNComponents(text: string): number | 'mle' | null {
  const t = text.trim();
  if (t === '') {
    return null;
  }
  if (t === 'mle') {
    return 'mle';
  }
  const v = Number(t);
  if (Number.isInteger(v) && v >= 1) {
    return v;
  }
  if (Number.isFinite(v) && v > 0 && v < 1) {
    return v;
  }
  throw new Error(
    `nComponents must be a positive integer, a fraction in (0,1), or 'mle' — got "${text}"`,
  );
}

function parseIteratedPower(text: string): number | 'auto' {
  const t = text.trim();
  if (t === '' || t === 'auto') {
    return 'auto';
  }
  const v = Number(t);
  if (Number.isInteger(v) && v >= 0) {
    return v;
  }
  throw new Error(`iteratedPower must be 'auto' or a non-negative integer — got "${text}"`);
}

function readConfig(): RunConfig {
  const seedText = inputEl('cfg-seed').value.trim();
  return {
    estimator: readRadio('estimator') as RunConfig['estimator'],
    mode: readRadio('mode') as RunConfig['mode'],
    backend: readRadio('backend') as RunConfig['backend'],
    nComponents: parseNComponents(inputEl('cfg-ncomponents').value),
    whiten: inputEl('cfg-whiten').checked,
    solver: byId<HTMLSelectElement>('cfg-solver').value as SvdSolver,
    iteratedPower: parseIteratedPower(inputEl('cfg-iterpower').value),
    seed: seedText === '' ? null : Number(seedText) >>> 0,
    batchSize: Math.max(1, Math.floor(Number(inputEl('cfg-batchsize').value) || 200)),
  };
}

function refreshConfigUI(): void {
  const est = readRadio('estimator');
  const mode = readRadio('mode');
  const ipca = est === 'ipca';
  byId('batchsize-label').hidden = !ipca;
  byId<HTMLSelectElement>('cfg-solver').disabled = ipca;
  inputEl('cfg-iterpower').disabled = ipca;
  inputEl('cfg-seed').disabled = ipca;
  const gpuAllowed = !ipca && mode === 'worker';
  setRadioDisabled('backend', !gpuAllowed);
  if (!gpuAllowed) {
    const cpu = document.querySelector<HTMLInputElement>('input[name="backend"][value="cpu"]');
    if (cpu !== null) {
      cpu.checked = true;
    }
  }
  const sel = readRadio('dataset');
  byId('synthetic-controls').hidden = sel !== 'synthetic';
  if (isPresetKind(sel)) {
    // The fixed presets have small p; lower an out-of-range integer nComponents
    // so the default "16" doesn't make every fit fail on the 3-D presets.
    const nc = inputEl('cfg-ncomponents');
    const v = Number(nc.value);
    if (Number.isInteger(v) && v > PRESET_INFO[sel].p) {
      nc.value = `${PRESET_INFO[sel].p}`;
    }
  }
  updateDataStatus();
}

function updateDataStatus(): void {
  const dtype = readRadio('dtype');
  const sel = readRadio('dataset');
  if (sel === 'digits') {
    setStatus('data-status', `sklearn digits: 1797×64 (8×8 images, values 0–16), ${dtype}`);
  } else if (isPresetKind(sel)) {
    const info = PRESET_INFO[sel];
    setStatus('data-status', `${info.blurb} — ${info.n}×${info.p}, ${dtype} (generated on Fit)`);
  } else {
    const n = inputEl('synth-n').value;
    const p = inputEl('synth-p').value;
    const c = inputEl('synth-clusters').value;
    setStatus(
      'data-status',
      `synthetic: ${n}×${p}, ${c} Gaussian clusters, ${dtype} (generated on Fit)`,
    );
  }
}

onRadioChange('estimator', refreshConfigUI);
onRadioChange('mode', refreshConfigUI);
onRadioChange('dataset', refreshConfigUI);
onRadioChange('dtype', updateDataStatus);
for (const id of ['synth-n', 'synth-p', 'synth-clusters']) {
  inputEl(id).addEventListener('input', updateDataStatus);
}

// ---------------------------------------------------------------------
// Dataset cache
// ---------------------------------------------------------------------

let dataset: Dataset | null = null;

async function ensureDataset(): Promise<Dataset> {
  const dtype = readRadio('dtype') as DemoDtype;
  const sel = readRadio('dataset');
  if (sel === 'digits') {
    if (dataset?.key !== `digits|${dtype}`) {
      dataset = await loadDigits(dtype);
    }
    return dataset;
  }
  if (isPresetKind(sel)) {
    const key = `${sel}|${dtype}`;
    if (dataset?.key !== key) {
      dataset = PRESET_INFO[sel].make(dtype);
    }
    return dataset;
  }
  const spec = {
    n: Number(inputEl('synth-n').value),
    p: Number(inputEl('synth-p').value),
    clusters: Number(inputEl('synth-clusters').value),
    dtype,
  };
  const key = `synthetic|${dtype}|${spec.n}|${spec.p}|${spec.clusters}`;
  if (dataset?.key !== key) {
    setStatus('data-status', `generating ${spec.n}×${spec.p}…`);
    await new Promise((r) => setTimeout(r, 0));
    dataset = makeSynthetic(spec);
    updateDataStatus();
  }
  return dataset;
}

// ---------------------------------------------------------------------
// Sessions: a uniform async facade over the four estimator flavors
// ---------------------------------------------------------------------

interface Session {
  kind: 'pca' | 'ipca';
  isWorker: boolean;
  /** Dataset the model was fitted on (null when restored without one). */
  data: Dataset | null;
  backendLabel: string;
  wallMs: number | null;
  k: number;
  p: number;
  components: Matrix;
  evr: FloatArray;
  embedding: Matrix | null;
  transform(X: MatrixInput): Promise<Matrix>;
  inverseTransform(Y: MatrixInput): Promise<Matrix>;
  scoreSamples: ((X: MatrixInput) => Promise<FloatArray>) | null;
  exportModel(): AnyPCAModel;
}

type SessionMeta = Pick<Session, 'data' | 'backendLabel' | 'wallMs'>;

function sessionFromPCA(est: PCA, meta: SessionMeta): Session {
  return {
    kind: 'pca',
    isWorker: false,
    ...meta,
    k: est.nComponents,
    p: est.nFeaturesIn,
    components: est.components,
    evr: est.explainedVarianceRatio,
    embedding: null,
    transform: async (X) => est.transform(X),
    inverseTransform: async (Y) => est.inverseTransform(Y),
    scoreSamples: async (X) => est.scoreSamples(X),
    exportModel: () => est.toModel(),
  };
}

function sessionFromIPCA(est: IncrementalPCA, meta: SessionMeta): Session {
  return {
    kind: 'ipca',
    isWorker: false,
    ...meta,
    k: est.nComponents,
    p: est.nFeaturesIn,
    components: est.components,
    evr: est.explainedVarianceRatio,
    embedding: null,
    transform: async (X) => est.transform(X),
    inverseTransform: async (Y) => est.inverseTransform(Y),
    scoreSamples: null,
    exportModel: () => est.toModel(),
  };
}

function sessionFromWorkerPCA(est: WorkerPCA, meta: SessionMeta): Session {
  return {
    kind: 'pca',
    isWorker: true,
    ...meta,
    k: est.nComponents,
    p: est.nFeaturesIn,
    components: est.components,
    evr: est.explainedVarianceRatio,
    embedding: null,
    transform: (X) => est.transform(X),
    inverseTransform: (Y) => est.inverseTransform(Y),
    scoreSamples: (X) => est.scoreSamples(X),
    exportModel: () => est.exportModel(),
  };
}

function sessionFromWorkerIPCA(est: WorkerIncrementalPCA, meta: SessionMeta): Session {
  return {
    kind: 'ipca',
    isWorker: true,
    ...meta,
    k: est.nComponents,
    p: est.nFeaturesIn,
    components: est.components,
    evr: est.explainedVarianceRatio,
    embedding: null,
    transform: (X) => est.transform(X),
    inverseTransform: (Y) => est.inverseTransform(Y),
    scoreSamples: null,
    exportModel: () => est.exportModel(),
  };
}

let session: Session | null = null;

// ---------------------------------------------------------------------
// The shared worker (one module worker reused across runs)
// ---------------------------------------------------------------------

let sharedWorker: Worker | null = null;
let disposePrevWorkerEst: (() => void) | null = null;

function getWorker(): Worker {
  sharedWorker ??= new Worker(new URL('./worker-entry.ts', import.meta.url), { type: 'module' });
  return sharedWorker;
}

function trackWorkerEst(est: WorkerPCA | WorkerIncrementalPCA): void {
  disposePrevWorkerEst?.();
  disposePrevWorkerEst = () => {
    est.dispose().catch(() => {});
  };
}

// ---------------------------------------------------------------------
// Progress plumbing
// ---------------------------------------------------------------------

function makeProgressHandler(data: Dataset): (e: PCAFitProgress) => void {
  let lastDraw = 0;
  return (e) => {
    let label = `${e.phase} ${e.step}${e.totalSteps !== null ? `/${e.totalSteps}` : ''}`;
    if (e.detail?.maxResidual !== undefined) {
      label += ` · residual ${e.detail.maxResidual.toExponential(1)}`;
    }
    setProgress(e.fraction, label);
    if (e.snapshot?.scores !== undefined) {
      const now = performance.now();
      if (now - lastDraw > 80 || e.phase === 'finalize') {
        lastDraw = now;
        scatter.update(e.snapshot.scores, data.labels, undefined, data.palette);
      }
    }
  };
}

// ---------------------------------------------------------------------
// Fit runners
// ---------------------------------------------------------------------

async function runPCA(cfg: RunConfig, data: Dataset, signal: AbortSignal): Promise<Session> {
  const options: PCAOptions = {
    nComponents: cfg.nComponents,
    whiten: cfg.whiten,
    svdSolver: cfg.solver,
    iteratedPower: cfg.iteratedPower,
    randomState: cfg.seed,
  };
  const onProgress = makeProgressHandler(data);
  if (cfg.mode === 'worker') {
    const est = new WorkerPCA({
      ...options,
      randomState: cfg.seed,
      backend: cfg.backend,
      worker: getWorker(),
    });
    trackWorkerEst(est);
    const fitOptions: WorkerFitOptions = {
      signal,
      onProgress,
      progress: { minIntervalMs: 50, snapshot: { scores: true } },
    };
    await est.fit(data.X, fitOptions);
    const info = await est.info();
    let backendLabel: string;
    if (info.backend === 'webgpu') {
      backendLabel = `WebGPU in worker (${info.gpuAdapterInfo ?? 'adapter unknown'})`;
    } else if (cfg.backend === 'webgpu') {
      backendLabel = info.webgpuAvailable
        ? 'CPU in worker (input below the GPU threshold)'
        : 'CPU in worker (WebGPU unavailable — transparent fallback)';
    } else {
      backendLabel = 'CPU in worker';
    }
    backendLabel += ` · solver ${est.resolvedSvdSolver}`;
    return sessionFromWorkerPCA(est, { data, backendLabel, wallMs: null });
  }
  const est = new PCA(options);
  if (cfg.mode === 'async') {
    await est.fitAsync(data.X, { signal, onProgress, snapshot: { scores: true }, budgetMs: 12 });
    return sessionFromPCA(est, {
      data,
      backendLabel: `CPU on main thread (fitAsync) · solver ${est.resolvedSvdSolver}`,
      wallMs: null,
    });
  }
  // Blocking fit: give the browser one frame to paint the running state,
  // then freeze — the liveness widget makes the freeze visible.
  await new Promise((r) => setTimeout(r, 30));
  est.fit(data.X, { onProgress, snapshot: { scores: true } });
  return sessionFromPCA(est, {
    data,
    backendLabel: `CPU on main thread (blocking fit — UI froze) · solver ${est.resolvedSvdSolver}`,
    wallMs: null,
  });
}

/** Contiguous [start, end) batches; a too-small tail is merged into the previous batch. */
function batchBounds(n: number, size: number, minLast: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let s0 = 0; s0 < n; s0 += size) {
    out.push([s0, Math.min(n, s0 + size)]);
  }
  const last = out[out.length - 1];
  if (out.length > 1 && last[1] - last[0] < minLast) {
    out.pop();
    out[out.length - 1][1] = n;
  }
  return out;
}

async function runIPCA(cfg: RunConfig, data: Dataset, signal: AbortSignal): Promise<Session> {
  if (cfg.nComponents !== null && typeof cfg.nComponents !== 'number') {
    throw new Error(
      "IncrementalPCA takes an integer (or empty) nComponents — not 'mle' or a fraction",
    );
  }
  const k = cfg.nComponents;
  if (k !== null && (!Number.isInteger(k) || k > Math.min(cfg.batchSize, data.p))) {
    throw new Error(
      `IncrementalPCA needs integer nComponents ≤ min(batchSize, nFeatures) = ${Math.min(cfg.batchSize, data.p)}`,
    );
  }
  const options: IncrementalPCAOptions = {
    nComponents: k,
    whiten: cfg.whiten,
    batchSize: cfg.batchSize,
  };
  let est: IncrementalPCA | WorkerIncrementalPCA;
  if (cfg.mode === 'worker') {
    const w = new WorkerIncrementalPCA({ ...options, worker: getWorker() });
    trackWorkerEst(w);
    est = w;
  } else {
    est = new IncrementalPCA(options);
  }
  const bounds = batchBounds(data.n, cfg.batchSize, k ?? 1);
  const seenEl = byId('ipca-seen');
  seenEl.hidden = false;
  let lastDraw = 0;
  let batchesDone = 0;
  try {
    for (let b = 0; b < bounds.length; b++) {
      if (signal.aborted) {
        throw signal.reason ?? new DOMException('The operation was aborted', 'AbortError');
      }
      const [s0, s1] = bounds[b];
      const batch = new Matrix(data.X.data.subarray(s0 * data.p, s1 * data.p), s1 - s0, data.p);
      await est.partialFit(batch);
      batchesDone++;
      setProgress((b + 1) / bounds.length, `batch ${b + 1}/${bounds.length} (partialFit)`);
      seenEl.textContent = `nSamplesSeen: ${est.nSamplesSeen}`;
      const now = performance.now();
      if (now - lastDraw > 300 || b === bounds.length - 1) {
        lastDraw = now;
        scatter.update(await est.transform(data.X), data.labels, undefined, data.palette);
      }
      if (cfg.mode === 'async') {
        await new Promise((r) => setTimeout(r, 0));
      }
    }
  } catch (err) {
    // IncrementalPCA keeps the completed-batch model on abort — show it.
    if ((err as Error | null)?.name === 'AbortError' && batchesDone > 0) {
      setStatus(
        'run-status',
        `aborted after ${batchesDone}/${bounds.length} batches — the completed-batch model is kept (IncrementalPCA contract)`,
      );
    } else {
      throw err;
    }
  }
  const where =
    cfg.mode === 'worker'
      ? 'CPU in worker'
      : cfg.mode === 'async'
        ? 'CPU on main thread (yielding between batches)'
        : 'CPU on main thread (blocking)';
  const meta = {
    data,
    backendLabel: `${where} · ${batchesDone} partialFit batches`,
    wallMs: null,
  };
  return est instanceof IncrementalPCA
    ? sessionFromIPCA(est, meta)
    : sessionFromWorkerIPCA(est, meta);
}

// ---------------------------------------------------------------------
// Run orchestration
// ---------------------------------------------------------------------

let running = false;
let abortCtl: AbortController | null = null;

function setRunning(on: boolean, mode: string): void {
  running = on;
  byId<HTMLButtonElement>('btn-run').disabled = on;
  byId<HTMLButtonElement>('btn-abort').disabled = !on || mode === 'sync';
  byId<HTMLFieldSetElement>('data-controls').disabled = on;
  byId<HTMLFieldSetElement>('config-controls').disabled = on;
}

async function runFit(): Promise<void> {
  if (running) {
    return;
  }
  let cfg: RunConfig;
  try {
    cfg = readConfig();
  } catch (err) {
    setStatus('run-status', (err as Error).message, 'err');
    return;
  }
  setStatus('run-status', '');
  setStatus('backend-line', '');
  byId('ipca-seen').hidden = true;
  byId('panel-results').hidden = true;
  scatter.clear();
  setRunning(true, cfg.mode);
  abortCtl = new AbortController();
  const t0 = performance.now();
  try {
    const data = await ensureDataset();
    setProgress(0, 'starting…');
    const s =
      cfg.estimator === 'pca'
        ? await runPCA(cfg, data, abortCtl.signal)
        : await runIPCA(cfg, data, abortCtl.signal);
    s.wallMs = performance.now() - t0;
    session = s;
    setProgress(1, 'done');
    setStatus('backend-line', `ran on ${s.backendLabel} — ${fmtMs(s.wallMs)}`, 'ok');
    addRunRow([
      cfg.estimator === 'pca' ? `PCA/${cfg.solver}` : 'IncrementalPCA',
      `${data.n}×${data.p} ${data.dtype}`,
      cfg.mode,
      s.backendLabel,
      fmtMs(s.wallMs),
    ]);
    await renderResults(s);
  } catch (err) {
    const e = err as Error;
    if (e?.name === 'AbortError') {
      setProgress(0, 'aborted');
      setStatus(
        'run-status',
        cfg.estimator === 'pca'
          ? 'aborted — the estimator is left unfitted (PCA contract)'
          : 'aborted before any batch completed',
        'err',
      );
    } else {
      console.error(err);
      setProgress(0, 'failed');
      setStatus('run-status', `${e?.name ?? 'Error'}: ${e?.message ?? String(err)}`, 'err');
    }
  } finally {
    setRunning(false, cfg.mode);
    abortCtl = null;
  }
}

byId('btn-run').addEventListener('click', () => {
  void runFit();
});
byId('btn-abort').addEventListener('click', () => {
  abortCtl?.abort();
});

// ---------------------------------------------------------------------
// Results panel
// ---------------------------------------------------------------------

interface ReconState {
  session: Session;
  Xsub: Matrix;
  Ysub: Matrix;
  captions: string[];
  side: number | null;
}

let reconState: ReconState | null = null;

async function renderResults(s: Session): Promise<void> {
  const panel = byId('panel-results');
  if (panel.hidden) {
    panel.hidden = false;
    resizeAllCharts();
  }
  scree.update(s.evr);
  const dataMatches = s.data !== null && s.data.p === s.p;
  if (dataMatches && s.data !== null) {
    s.embedding = await s.transform(s.data.X);
    scatter.update(s.embedding, s.data.labels, undefined, s.data.palette);
  }
  renderEigenTiles(s);
  await setupReconstruction(s, dataMatches ? s.data : null);
  await renderOutliers(s, dataMatches ? s.data : null);
}

function renderEigenTiles(s: Session): void {
  const block = byId('eigen-block');
  const side = Math.round(Math.sqrt(s.p));
  const isImage = side * side === s.p && side === 8;
  block.hidden = !isImage;
  if (!isImage) {
    return;
  }
  const count = Math.min(s.k, 16);
  const tiles = [];
  for (let i = 0; i < count; i++) {
    tiles.push({
      values: s.components.data.subarray(i * s.p, (i + 1) * s.p),
      caption: `PC${i + 1} · ${(s.evr[i] * 100).toFixed(1)}%`,
    });
  }
  renderTiles(byId('eigen-tiles'), tiles, side, 'diverging');
}

async function setupReconstruction(s: Session, data: Dataset | null): Promise<void> {
  const block = byId('recon-block');
  reconState = null;
  if (data === null || s.k < 1) {
    block.hidden = true;
    return;
  }
  block.hidden = false;
  // Sample rows: for digits, the first instance of each class; otherwise the first 10 rows.
  const idx: number[] = [];
  if (data.kind === 'digits') {
    const seen = new Set<number>();
    for (let i = 0; i < data.n && seen.size < 10; i++) {
      if (!seen.has(data.labels[i])) {
        seen.add(data.labels[i]);
        idx.push(i);
      }
    }
    idx.sort((a, b) => data.labels[a] - data.labels[b]);
  } else {
    for (let i = 0; i < Math.min(10, data.n); i++) {
      idx.push(i);
    }
  }
  const sub = new Float64Array(idx.length * data.p);
  for (let r = 0; r < idx.length; r++) {
    sub.set(data.X.data.subarray(idx[r] * data.p, (idx[r] + 1) * data.p), r * data.p);
  }
  const Xsub = new Matrix(sub, idx.length, data.p);
  const Ysub = await s.transform(Xsub);
  const captions = idx.map((i) =>
    data.kind === 'digits' ? `digit ${data.labels[i]}` : `row ${i}`,
  );
  reconState = { session: s, Xsub, Ysub, captions, side: data.imageSide };
  if (data.imageSide !== null) {
    const originals = captions.map((caption, r) => ({
      values: Xsub.data.subarray(r * data.p, (r + 1) * data.p),
      caption,
    }));
    renderTiles(byId('recon-orig'), originals, data.imageSide, 'ink', 5);
  } else {
    byId('recon-orig').textContent = '';
    byId('recon-tiles').textContent = '';
  }
  const slider = inputEl('recon-k');
  slider.max = `${s.k}`;
  slider.value = `${Math.min(s.k, Math.max(1, Math.round(s.k / 2)))}`;
  byId('recon-k-out').textContent = slider.value;
  await updateReconstruction();
  await renderMseCurve(s, data);
}

/** Truncating to k' = zeroing trailing score columns; exact for whitened fits too. */
function truncateScores(Y: Matrix, keep: number): Matrix {
  const data = Y.data.slice();
  for (let r = 0; r < Y.rows; r++) {
    data.fill(0, r * Y.cols + keep, (r + 1) * Y.cols);
  }
  return new Matrix(data, Y.rows, Y.cols);
}

function mseBetween(a: Matrix, b: Matrix): number {
  let acc = 0;
  for (let i = 0; i < a.data.length; i++) {
    const d = a.data[i] - b.data[i];
    acc += d * d;
  }
  return acc / a.data.length;
}

async function updateReconstruction(): Promise<void> {
  const st = reconState;
  if (st === null) {
    return;
  }
  const keep = Number(inputEl('recon-k').value);
  try {
    const R = await st.session.inverseTransform(truncateScores(st.Ysub, keep));
    if (st.session !== reconState?.session) {
      return; // a newer fit replaced this session while we awaited
    }
    byId('recon-mse').textContent = `MSE @ k=${keep}: ${mseBetween(R, st.Xsub).toPrecision(3)}`;
    if (st.side !== null) {
      const tiles = st.captions.map((caption, r) => ({
        values: R.data.subarray(r * R.cols, (r + 1) * R.cols),
        caption,
      }));
      renderTiles(byId('recon-tiles'), tiles, st.side, 'ink', 5);
    }
  } catch {
    // stale session (disposed worker estimator) — a rerun will rebind
  }
}

async function renderMseCurve(s: Session, data: Dataset): Promise<void> {
  const m = Math.min(200, data.n);
  const Xm = new Matrix(data.X.data.subarray(0, m * data.p), m, data.p);
  const Ym = await s.transform(Xm);
  const ks: number[] = [];
  const step = Math.max(1, Math.ceil(s.k / 16));
  for (let k = 1; k <= s.k; k += step) {
    ks.push(k);
  }
  if (ks[ks.length - 1] !== s.k) {
    ks.push(s.k);
  }
  const mses: number[] = [];
  for (const k of ks) {
    const R = await s.inverseTransform(truncateScores(Ym, k));
    mses.push(mseBetween(R, Xm));
  }
  if (session === s) {
    mseLine.update(ks, mses);
  }
}

let reconRaf = 0;
inputEl('recon-k').addEventListener('input', () => {
  cancelAnimationFrame(reconRaf);
  reconRaf = requestAnimationFrame(() => {
    void updateReconstruction();
  });
});

async function renderOutliers(s: Session, data: Dataset | null): Promise<void> {
  const block = byId('outlier-block');
  if (s.scoreSamples === null || data === null) {
    block.hidden = true;
    return;
  }
  block.hidden = false;
  try {
    const raw = await s.scoreSamples(data.X);
    const vals = Float64Array.from(raw);
    const finite = vals.filter((v) => Number.isFinite(v));
    if (finite.length === 0) {
      throw new Error('all log-likelihoods are non-finite (is nComponents = nFeatures?)');
    }
    const sorted = finite.slice().sort((a, b) => a - b);
    const thr = sorted[Math.floor(0.02 * (sorted.length - 1))];
    hist.update(finite, thr);
    const outliers = new Set<number>();
    for (let i = 0; i < vals.length; i++) {
      if (!(vals[i] > thr)) {
        outliers.add(i); // includes NaN/−∞ rows
      }
    }
    setStatus(
      'outlier-note',
      `${outliers.size} samples at or below the 2nd-percentile log-likelihood (${thr.toFixed(1)}) — ringed in the embedding above`,
    );
    if (s.embedding !== null) {
      scatter.update(s.embedding, data.labels, outliers, data.palette);
    }
  } catch (err) {
    setStatus('outlier-note', `scoreSamples unavailable: ${(err as Error).message}`, 'err');
  }
}

// ---------------------------------------------------------------------
// Model persistence: JSON download/upload and IndexedDB (structured clone)
// ---------------------------------------------------------------------

function describeModel(m: AnyPCAModel): string {
  const name = m.estimator === 'ipca' ? 'IncrementalPCA' : 'PCA';
  return `${name} · ${m.nComponents} components × ${m.nFeaturesIn} features`;
}

async function restoreFromModel(model: AnyPCAModel, source: string): Promise<void> {
  const est =
    model.estimator === 'ipca'
      ? IncrementalPCA.fromModel(model as IncrementalPCAModel)
      : PCA.fromModel(model as PCAModel);
  let data: Dataset | null = null;
  try {
    const d = await ensureDataset();
    data = d.p === model.nFeaturesIn ? d : null;
  } catch {
    data = null;
  }
  const meta = {
    data,
    backendLabel: `restored from ${source} — no refit`,
    wallMs: null,
  };
  session = est instanceof IncrementalPCA ? sessionFromIPCA(est, meta) : sessionFromPCA(est, meta);
  setStatus('backend-line', `model ${meta.backendLabel}`, 'ok');
  setStatus(
    'model-status',
    `${describeModel(model)} restored from ${source}${data === null ? ' (current dataset has a different feature count — data-dependent panels hidden)' : ''}`,
    'ok',
  );
  await renderResults(session);
}

function idbOpen(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('pca-web-demo', 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore('models');
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('IndexedDB open failed'));
  });
}

async function idbPut(model: AnyPCAModel): Promise<void> {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('models', 'readwrite');
    tx.objectStore('models').put(model, 'last');
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = () => {
      db.close();
      reject(tx.error ?? new Error('IndexedDB write failed'));
    };
  });
}

async function idbGet(): Promise<AnyPCAModel | undefined> {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const req = db.transaction('models', 'readonly').objectStore('models').get('last');
    req.onsuccess = () => {
      db.close();
      resolve(req.result as AnyPCAModel | undefined);
    };
    req.onerror = () => {
      db.close();
      reject(req.error ?? new Error('IndexedDB read failed'));
    };
  });
}

byId('btn-export').addEventListener('click', () => {
  if (session === null) {
    setStatus('model-status', 'fit (or import) a model first', 'err');
    return;
  }
  const model = session.exportModel();
  download(`pca-web-${model.estimator}-model.json`, modelToJSON(model));
  setStatus('model-status', `${describeModel(model)} exported as JSON`, 'ok');
});

inputEl('file-import').addEventListener('change', () => {
  const file = inputEl('file-import').files?.[0];
  if (file === undefined) {
    return;
  }
  file
    .text()
    .then((text) => restoreFromModel(modelFromJSON(text), 'imported JSON'))
    .catch((err) => setStatus('model-status', `import failed: ${(err as Error).message}`, 'err'));
  inputEl('file-import').value = '';
});

byId('btn-idb-save').addEventListener('click', () => {
  if (session === null) {
    setStatus('model-status', 'fit (or import) a model first', 'err');
    return;
  }
  const model = session.exportModel();
  idbPut(model)
    .then(() =>
      setStatus(
        'model-status',
        `${describeModel(model)} saved to IndexedDB (structured clone — typed arrays stored directly). Reload the page and press “Load from IndexedDB”.`,
        'ok',
      ),
    )
    .catch((err) => setStatus('model-status', `save failed: ${(err as Error).message}`, 'err'));
});

byId('btn-idb-load').addEventListener('click', () => {
  idbGet()
    .then((model) => {
      if (model === undefined) {
        setStatus('model-status', 'no saved model in IndexedDB yet', 'err');
        return;
      }
      return restoreFromModel(model, 'IndexedDB');
    })
    .catch((err) => setStatus('model-status', `load failed: ${(err as Error).message}`, 'err'));
});

// ---------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------

refreshConfigUI();
idbGet()
  .then((model) => {
    if (model !== undefined && session === null) {
      setStatus(
        'model-status',
        `IndexedDB holds a saved ${describeModel(model)} — “Load from IndexedDB” restores it without refitting.`,
      );
    }
  })
  .catch(() => {});
