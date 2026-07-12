/**
 * Cooperative scheduling for non-blocking fits: abort signalling, event-loop
 * yielding, time-slice budgeting, and the two generator drivers (synchronous
 * drain and time-sliced async). Environment-agnostic — browser main thread,
 * Web Workers, and Node all work; every host global is feature-detected
 * through `globalThis` (the build targets lib ES2022 only).
 */

/**
 * Structural subset of `AbortSignal` read by the fit drivers, so any object
 * with an `aborted` flag works (no DOM lib required).
 */
export interface AbortSignalLike {
  readonly aborted: boolean;
  readonly reason?: unknown;
}

/**
 * Thrown (or rejected) when a fit is aborted through a signal that carries
 * no `reason`. `name` is 'AbortError', matching the DOMException convention,
 * so `err.name === 'AbortError'` identifies cancellations portably.
 */
export class FitAbortError extends Error {
  override readonly name = 'AbortError';

  constructor(message = 'The operation was aborted') {
    super(message);
  }
}

/** Throws `signal.reason` (or a `FitAbortError`) when the signal is aborted. */
export function throwIfAborted(signal?: AbortSignalLike | null): void {
  if (signal?.aborted) {
    throw signal.reason ?? new FitAbortError();
  }
}

interface SchedulingGlobals {
  scheduler?: { yield?: () => Promise<void> };
  setImmediate?: (cb: () => void) => unknown;
  MessageChannel?: new () => {
    port1: { onmessage: ((ev: unknown) => void) | null };
    port2: { postMessage(value: unknown): void };
  };
  setTimeout?: (cb: () => void, ms?: number) => unknown;
  performance?: { now(): number };
}

const GLOBALS = globalThis as SchedulingGlobals;

let yieldImpl: (() => Promise<void>) | null = null;

/** Picks the best macrotask-yield mechanism available — once, lazily. */
function pickYieldImpl(): () => Promise<void> {
  const scheduler = GLOBALS.scheduler;
  if (scheduler && typeof scheduler.yield === 'function') {
    const schedulerYield = scheduler.yield.bind(scheduler);
    return () => schedulerYield();
  }
  if (typeof GLOBALS.setImmediate === 'function') {
    const setImmediateFn = GLOBALS.setImmediate;
    return () => new Promise((resolve) => setImmediateFn(resolve));
  }
  if (typeof GLOBALS.MessageChannel === 'function') {
    // One long-lived channel; each posted message wakes one waiter (FIFO).
    // Unlike setTimeout(0), this is not throttled to >=4ms by browsers.
    const channel = new GLOBALS.MessageChannel();
    const waiters: Array<() => void> = [];
    channel.port1.onmessage = () => {
      waiters.shift()?.();
    };
    return () =>
      new Promise((resolve) => {
        waiters.push(resolve);
        channel.port2.postMessage(null);
      });
  }
  if (typeof GLOBALS.setTimeout === 'function') {
    const setTimeoutFn = GLOBALS.setTimeout;
    return () => new Promise((resolve) => setTimeoutFn(resolve, 0));
  }
  // Hosts without timers: a microtask still lets chained promises interleave.
  return () => Promise.resolve();
}

/**
 * Resolves in a later macrotask, giving the host a chance to process input,
 * rendering, and messages. Prefers `scheduler.yield()` (Chrome 129+), then
 * `setImmediate` (Node), then a `MessageChannel` round-trip, then
 * `setTimeout(0)`.
 */
export function yieldToEventLoop(): Promise<void> {
  yieldImpl ??= pickYieldImpl();
  return yieldImpl();
}

function defaultNow(): () => number {
  const perf = GLOBALS.performance;
  if (perf && typeof perf.now === 'function') {
    return () => perf.now();
  }
  return () => Date.now();
}

/**
 * Tracks a wall-clock budget per event-loop slice. `pause()` returns null
 * while the current slice has budget left — callers skip awaiting entirely,
 * avoiding microtask churn in the hot loop — or a promise that yields to the
 * event loop and restarts the budget.
 */
export class TimeSlicer {
  private readonly budgetMs: number;
  private readonly now: () => number;
  private sliceStart: number;

  constructor(budgetMs: number, nowFn?: () => number) {
    this.budgetMs = budgetMs;
    this.now = nowFn ?? defaultNow();
    this.sliceStart = this.now();
  }

  pause(): Promise<void> | null {
    if (this.now() - this.sliceStart < this.budgetMs) {
      return null;
    }
    return yieldToEventLoop().then(() => {
      this.sliceStart = this.now();
    });
  }
}

/**
 * Drains a step generator synchronously — the classic blocking fit. The
 * abort signal is checked before every step; on any exit before completion
 * (abort, callback throw) the generator's `finally` blocks run via
 * `gen.return`, releasing estimator state such as the re-entrancy flag.
 */
export function driveSync<T>(gen: Generator<void, T, void>, signal?: AbortSignalLike | null): T {
  let finished = false;
  try {
    for (;;) {
      throwIfAborted(signal);
      const step = gen.next();
      if (step.done) {
        finished = true;
        return step.value;
      }
    }
  } finally {
    if (!finished) {
      gen.return(undefined as unknown as T);
    }
  }
}

export interface DriveAsyncOptions {
  /** Milliseconds of solver work per event-loop slice (default 12). */
  budgetMs?: number;
  signal?: AbortSignalLike | null;
}

/**
 * Drives a step generator cooperatively: runs steps until the slice's time
 * budget is spent, then yields to the event loop. Yields once before the
 * first step so callers can paint (a spinner, a disabled button) ahead of
 * any heavy work. A `budgetMs` of 0 yields after every step.
 */
export async function driveAsync<T>(
  gen: Generator<void, T, void>,
  options: DriveAsyncOptions = {},
): Promise<T> {
  const signal = options.signal;
  let finished = false;
  try {
    throwIfAborted(signal);
    await yieldToEventLoop();
    const slicer = new TimeSlicer(options.budgetMs ?? 12);
    for (;;) {
      throwIfAborted(signal);
      const step = gen.next();
      if (step.done) {
        finished = true;
        return step.value;
      }
      const pause = slicer.pause();
      if (pause !== null) {
        await pause;
      }
    }
  } finally {
    if (!finished) {
      gen.return(undefined as unknown as T);
    }
  }
}
