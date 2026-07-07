/**
 * WebGPU execution engine: device acquisition with feature detection, ds
 * buffer management, and the exact-binned GEMM dispatch used by the
 * WebGPUPCA frontend (see kernels.ts for the numerical design).
 *
 * The engine verifies at creation time that the kernel actually reaches
 * near-f64 accuracy on the running driver; if the self-check fails,
 * creation reports no engine and callers use the CPU path — numerical
 * equivalence is part of the API contract.
 */
import type { FloatArray } from '../types.js';
import { encodeDs, maxAbs, prescaleExponent } from './df64.js';
import { BIN_LEVELS, gemmShader, MAX_K_PER_DISPATCH, OUT_SLOTS, TILE } from './kernels.js';

export interface WebGPUDeviceOptions {
  /** Bring your own device (it will not be destroyed by dispose()). */
  device?: GPUDevice;
  powerPreference?: GPUPowerPreference;
}

/** True when a WebGPU implementation is present (browser or runtime flag). */
export function isWebGPUSupported(): boolean {
  const nav = (globalThis as { navigator?: { gpu?: unknown } }).navigator;
  return typeof nav?.gpu === 'object' && nav.gpu !== null;
}

/** A device-resident ds matrix (prescaled by 2^-scaleExp). */
export interface GpuMat {
  readonly buffer: GPUBuffer;
  readonly rows: number;
  readonly cols: number;
  readonly scaleExp: number;
  destroy(): void;
}

interface GemmSpec {
  a: GpuMat;
  b: GpuMat;
  ta: boolean;
  tb: boolean;
  m: number;
  n: number;
  k: number;
}

export class GpuEngine {
  readonly device: GPUDevice;
  readonly adapterInfo: string;
  private readonly ownsDevice: boolean;
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  private disposed = false;
  private relError = Number.NaN;

  private constructor(device: GPUDevice, ownsDevice: boolean, adapterInfo: string) {
    this.device = device;
    this.ownsDevice = ownsDevice;
    this.adapterInfo = adapterInfo;
  }

  /** Achieved |C_gpu − C_cpu| / |C_cpu| in the creation-time GEMM self-check. */
  get measuredRelError(): number {
    return this.relError;
  }

  /**
   * Human-readable reason for the most recent create() returning null
   * (feature detection, adapter/device failure, or self-check details).
   */
  static lastCreateError: string | null = null;

  /**
   * Acquires a device and verifies GEMM precision. Returns null when WebGPU
   * is unavailable, the adapter/device request fails, or the precision
   * self-check does not reach near-f64 accuracy.
   */
  static async create(options: WebGPUDeviceOptions = {}): Promise<GpuEngine | null> {
    GpuEngine.lastCreateError = null;
    let device = options.device ?? null;
    const ownsDevice = device === null;
    let adapterInfo = 'injected device';
    if (device === null) {
      if (!isWebGPUSupported()) {
        GpuEngine.lastCreateError = 'navigator.gpu is not available';
        return null;
      }
      const gpu = (globalThis as unknown as { navigator: { gpu: GPU } }).navigator.gpu;
      let adapter: GPUAdapter | null = null;
      try {
        adapter = await gpu.requestAdapter({
          powerPreference: options.powerPreference ?? 'high-performance',
        });
      } catch (err) {
        GpuEngine.lastCreateError = `requestAdapter threw: ${String(err)}`;
        return null;
      }
      if (adapter === null) {
        GpuEngine.lastCreateError = 'requestAdapter returned null';
        return null;
      }
      const info = adapter.info;
      adapterInfo = [info?.vendor, info?.architecture, info?.device, info?.description]
        .filter((s) => typeof s === 'string' && s.length > 0)
        .join(' / ');
      try {
        device = await adapter.requestDevice({
          requiredLimits: {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
          },
        });
      } catch (err) {
        GpuEngine.lastCreateError = `requestDevice failed on "${adapterInfo}": ${String(err)}`;
        return null;
      }
    }
    const engine = new GpuEngine(device, ownsDevice, adapterInfo);
    try {
      engine.relError = await engine.precisionSelfCheck();
      // Plain f32 accumulation would sit around 1e-7; the exact-binned
      // kernel lands near 1e-14.
      if (!(engine.relError < 1e-10)) {
        GpuEngine.lastCreateError = `precision self-check too imprecise on "${adapterInfo}": relErr=${engine.relError.toExponential(2)}`;
        engine.dispose();
        return null;
      }
      return engine;
    } catch (err) {
      GpuEngine.lastCreateError = `precision self-check failed on "${adapterInfo}": ${String(err)}`;
      engine.dispose();
      return null;
    }
  }

  /** Whether an (rows×cols) ds matrix fits the device's binding limits. */
  canFit(rows: number, cols: number): boolean {
    const dsBytes = rows * cols * 8;
    const outBytes = rows * cols * OUT_SLOTS * 4;
    const lim = this.device.limits;
    const maxDim = Math.max(rows, cols);
    return (
      dsBytes <= lim.maxStorageBufferBindingSize &&
      outBytes <= lim.maxStorageBufferBindingSize &&
      outBytes <= lim.maxBufferSize &&
      Math.ceil(maxDim / TILE) <= lim.maxComputeWorkgroupsPerDimension
    );
  }

  /** Uploads a matrix to the device, prescaled and ds-encoded. */
  upload(data: FloatArray, rows: number, cols: number): GpuMat {
    const scaleExp = prescaleExponent(maxAbs(data));
    const ds = encodeDs(data, scaleExp);
    const buffer = this.device.createBuffer({
      size: ds.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, ds);
    return { buffer, rows, cols, scaleExp, destroy: () => buffer.destroy() };
  }

  /** C = A @ b for a device-resident A and a small CPU panel b (cols×w). */
  async mulA(a: GpuMat, b: Float64Array, w: number): Promise<Float64Array> {
    const panel = this.upload(b, a.cols, w);
    try {
      return await this.gemm({ a, b: panel, ta: false, tb: false, m: a.rows, n: w, k: a.cols });
    } finally {
      panel.destroy();
    }
  }

  /** C = Aᵀ @ b for a device-resident A and a small CPU panel b (rows×w). */
  async mulAT(a: GpuMat, b: Float64Array, w: number): Promise<Float64Array> {
    const panel = this.upload(b, a.rows, w);
    try {
      return await this.gemm({ a, b: panel, ta: true, tb: false, m: a.cols, n: w, k: a.rows });
    } finally {
      panel.destroy();
    }
  }

  /** C = bᵀ @ A (w×cols) for a device-resident A and a small CPU panel b (rows×w). */
  async mulTA(a: GpuMat, b: Float64Array, w: number): Promise<Float64Array> {
    const panel = this.upload(b, a.rows, w);
    try {
      return await this.gemm({
        a: panel,
        b: a,
        ta: true,
        tb: false,
        m: w,
        n: a.cols,
        k: a.rows,
      });
    } finally {
      panel.destroy();
    }
  }

  /** Raw Gram matrix C = AᵀA (cols×cols) of a device-resident A. */
  async syrk(a: GpuMat): Promise<Float64Array> {
    return this.gemm({ a, b: a, ta: true, tb: false, m: a.cols, n: a.cols, k: a.rows });
  }

  /** Projection C = X @ compᵀ (n×k) with both operands device-resident. */
  async project(x: GpuMat, comp: GpuMat): Promise<Float64Array> {
    return this.gemm({ a: x, b: comp, ta: false, tb: true, m: x.rows, n: comp.rows, k: x.cols });
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.pipelines.clear();
    if (this.ownsDevice) {
      this.device.destroy();
    }
  }

  // ------------------------------------------------------------------

  private pipeline(ta: boolean, tb: boolean): GPUComputePipeline {
    const key = `${ta}-${tb}`;
    let p = this.pipelines.get(key);
    if (p === undefined) {
      const module = this.device.createShaderModule({ code: gemmShader(ta, tb) });
      p = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  /**
   * Runs op(A)·op(B) with exact integer-binned accumulation, splitting the
   * K dimension when it exceeds the per-dispatch overflow bound; chunk
   * results are summed in f64. Returns plain f64 values (rescaled).
   */
  private async gemm(spec: GemmSpec): Promise<Float64Array> {
    const out = new Float64Array(spec.m * spec.n);
    // Grid exponent: prescaled inputs are ≤ 1 in magnitude, so every
    // product term is < 2^1; +2 gives headroom for the bound being lax.
    const e = 2;
    const resultScale = 2 ** (spec.a.scaleExp + spec.b.scaleExp);
    for (let kStart = 0; kStart < spec.k; kStart += MAX_K_PER_DISPATCH) {
      const kCount = Math.min(MAX_K_PER_DISPATCH, spec.k - kStart);
      const bins = await this.dispatchChunk(spec, kStart, kCount, e);
      for (let i = 0; i < out.length; i++) {
        const base = i * OUT_SLOTS;
        let acc = 0;
        for (let lvl = 0; lvl < BIN_LEVELS; lvl++) {
          acc += bins[base + lvl] * 2 ** (e - 13 * (lvl + 1));
        }
        out[i] += acc;
      }
    }
    if (resultScale !== 1) {
      for (let i = 0; i < out.length; i++) {
        out[i] *= resultScale;
      }
    }
    return out;
  }

  private async dispatchChunk(
    spec: GemmSpec,
    kStart: number,
    kCount: number,
    e: number,
  ): Promise<Int32Array> {
    const { device } = this;
    const outBytes = spec.m * spec.n * OUT_SLOTS * 4;
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');

    const out = device.createBuffer({
      size: outBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const staging = device.createBuffer({
      size: outBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const dims = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const dimsData = new ArrayBuffer(32);
    const u32 = new Uint32Array(dimsData);
    const i32 = new Int32Array(dimsData);
    u32[0] = spec.m;
    u32[1] = spec.n;
    u32[2] = spec.k;
    u32[3] = kStart;
    u32[4] = kCount;
    i32[5] = e;
    device.queue.writeBuffer(dims, 0, dimsData);

    const pipeline = this.pipeline(spec.ta, spec.tb);
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: spec.a.buffer } },
        { binding: 1, resource: { buffer: spec.b.buffer } },
        { binding: 2, resource: { buffer: out } },
        { binding: 3, resource: { buffer: dims } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(spec.n / TILE), Math.ceil(spec.m / TILE));
    pass.end();
    encoder.copyBufferToBuffer(out, 0, staging, 0, outBytes);
    device.queue.submit([encoder.finish()]);

    try {
      const validationError = await device.popErrorScope();
      const oomError = await device.popErrorScope();
      if (validationError || oomError) {
        throw new Error(`WebGPU GEMM failed: ${(validationError ?? oomError)?.message}`);
      }
      await staging.mapAsync(GPUMapMode.READ);
      const bins = new Int32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return bins;
    } finally {
      out.destroy();
      staging.destroy();
      dims.destroy();
    }
  }

  /** Runs a small GEMM whose exact f64 result is computed on the CPU. */
  private async precisionSelfCheck(): Promise<number> {
    const m = 33;
    const k = 47;
    const n = 29;
    // Deterministic values with nontrivial low-order bits (exercise the lo
    // halves of the ds encoding and the deep bin levels).
    const a = new Float64Array(m * k);
    const b = new Float64Array(k * n);
    for (let i = 0; i < a.length; i++) {
      a[i] = Math.sin(i + 1) * 3 + 1e-9 * Math.cos(i * 7);
    }
    for (let i = 0; i < b.length; i++) {
      b[i] = Math.cos(i + 2) * 2 + 1e-9 * Math.sin(i * 5);
    }
    const ref = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let acc = 0;
        for (let l = 0; l < k; l++) {
          acc += a[i * k + l] * b[l * n + j];
        }
        ref[i * n + j] = acc;
      }
    }
    const ga = this.upload(a, m, k);
    const gb = this.upload(b, k, n);
    try {
      const got = await this.gemm({ a: ga, b: gb, ta: false, tb: false, m, n, k });
      let refMax = 0;
      for (let i = 0; i < ref.length; i++) {
        refMax = Math.max(refMax, Math.abs(ref[i]));
      }
      let worst = 0;
      for (let i = 0; i < ref.length; i++) {
        // Scale-aware denominator: a near-cancelled reference entry must not
        // spuriously fail the check (its absolute error is what matters).
        const denom = Math.max(Math.abs(ref[i]), 1e-3 * refMax);
        worst = Math.max(worst, Math.abs(got[i] - ref[i]) / denom);
      }
      return worst;
    } finally {
      ga.destroy();
      gb.destroy();
    }
  }
}
