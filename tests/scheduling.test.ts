import { describe, expect, it } from 'vitest';
import {
  driveAsync,
  driveSync,
  FitAbortError,
  TimeSlicer,
  throwIfAborted,
  yieldToEventLoop,
} from '../src/scheduling.js';

/** A generator that logs each step and flags whether its finally ran. */
function makeSteps(
  n: number,
  log: number[],
  flags: { finallyRan: boolean },
  onStep?: (i: number) => void,
): Generator<void, string, void> {
  return (function* steps(): Generator<void, string, void> {
    try {
      for (let i = 0; i < n; i++) {
        log.push(i);
        onStep?.(i);
        yield;
      }
      return 'done';
    } finally {
      flags.finallyRan = true;
    }
  })();
}

describe('throwIfAborted', () => {
  it('is a no-op for missing or unaborted signals', () => {
    expect(() => throwIfAborted()).not.toThrow();
    expect(() => throwIfAborted(null)).not.toThrow();
    expect(() => throwIfAborted({ aborted: false })).not.toThrow();
  });

  it('throws a FitAbortError with name AbortError when no reason is set', () => {
    try {
      throwIfAborted({ aborted: true });
      expect.unreachable('should have thrown');
    } catch (err) {
      expect(err).toBeInstanceOf(FitAbortError);
      expect((err as Error).name).toBe('AbortError');
    }
  });

  it('throws the signal reason when present', () => {
    const reason = new Error('custom cancellation');
    expect(() => throwIfAborted({ aborted: true, reason })).toThrow(reason);
  });

  it('works with a real AbortController signal', () => {
    const controller = new AbortController();
    expect(() => throwIfAborted(controller.signal)).not.toThrow();
    controller.abort();
    try {
      throwIfAborted(controller.signal);
      expect.unreachable('should have thrown');
    } catch (err) {
      expect((err as Error).name).toBe('AbortError');
    }
  });
});

describe('yieldToEventLoop', () => {
  it('resolves after letting queued macrotasks run', async () => {
    let macrotaskRan = false;
    setTimeout(() => {
      macrotaskRan = true;
    }, 0);
    // One yield may beat the timer (setImmediate runs before timers);
    // a few consecutive yields must let it through.
    for (let i = 0; i < 5 && !macrotaskRan; i++) {
      await yieldToEventLoop();
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(macrotaskRan).toBe(true);
  });
});

describe('TimeSlicer', () => {
  it('pauses only when the injected clock exceeds the budget, then restarts it', async () => {
    let t = 0;
    const slicer = new TimeSlicer(10, () => t);
    expect(slicer.pause()).toBeNull();
    t = 9;
    expect(slicer.pause()).toBeNull();
    t = 10;
    const pause = slicer.pause();
    expect(pause).not.toBeNull();
    await pause;
    // The budget restarted at t=10.
    t = 19;
    expect(slicer.pause()).toBeNull();
    t = 20;
    expect(slicer.pause()).not.toBeNull();
  });

  it('with a zero budget pauses at every step', () => {
    const slicer = new TimeSlicer(0, () => 0);
    expect(slicer.pause()).not.toBeNull();
  });
});

describe('driveSync', () => {
  it('drains the generator and returns its value', () => {
    const log: number[] = [];
    const flags = { finallyRan: false };
    const result = driveSync(makeSteps(3, log, flags));
    expect(result).toBe('done');
    expect(log).toEqual([0, 1, 2]);
    expect(flags.finallyRan).toBe(true);
  });

  it('throws before running any step on a pre-aborted signal', () => {
    const log: number[] = [];
    const flags = { finallyRan: false };
    expect(() => driveSync(makeSteps(3, log, flags), { aborted: true })).toThrow(FitAbortError);
    expect(log).toEqual([]);
  });

  it('on mid-run abort stops, throws AbortError, and runs the generator finally', () => {
    const log: number[] = [];
    const flags = { finallyRan: false };
    const signal = { aborted: false };
    const gen = makeSteps(10, log, flags, (i) => {
      if (i === 2) {
        signal.aborted = true;
      }
    });
    try {
      driveSync(gen, signal);
      expect.unreachable('should have thrown');
    } catch (err) {
      expect((err as Error).name).toBe('AbortError');
    }
    expect(log).toEqual([0, 1, 2]);
    expect(flags.finallyRan).toBe(true);
  });

  it('accepts a minimal structural signal object', () => {
    const log: number[] = [];
    const flags = { finallyRan: false };
    expect(driveSync(makeSteps(2, log, flags), { aborted: false })).toBe('done');
  });
});

describe('driveAsync', () => {
  it('resolves with the generator value', async () => {
    const log: number[] = [];
    const flags = { finallyRan: false };
    await expect(driveAsync(makeSteps(4, log, flags), { budgetMs: 0 })).resolves.toBe('done');
    expect(log).toEqual([0, 1, 2, 3]);
    expect(flags.finallyRan).toBe(true);
  });

  it('rejects before running any step on a pre-aborted signal', async () => {
    const log: number[] = [];
    const flags = { finallyRan: false };
    await expect(
      driveAsync(makeSteps(3, log, flags), { signal: { aborted: true } }),
    ).rejects.toMatchObject({ name: 'AbortError' });
    expect(log).toEqual([]);
  });

  it('on mid-run abort rejects and runs the generator finally', async () => {
    const log: number[] = [];
    const flags = { finallyRan: false };
    const controller = new AbortController();
    const gen = makeSteps(10, log, flags, (i) => {
      if (i === 2) {
        controller.abort();
      }
    });
    await expect(driveAsync(gen, { budgetMs: 0, signal: controller.signal })).rejects.toMatchObject(
      {
        name: 'AbortError',
      },
    );
    expect(log).toEqual([0, 1, 2]);
    expect(flags.finallyRan).toBe(true);
  });

  it('keeps the event loop live while stepping', async () => {
    let ticks = 0;
    const interval = setInterval(() => {
      ticks++;
    }, 0);
    const log: number[] = [];
    const flags = { finallyRan: false };
    // Each step burns ~1ms so timers actually come due between slices.
    const gen = makeSteps(25, log, flags, () => {
      const until = performance.now() + 1;
      while (performance.now() < until) {
        // spin
      }
    });
    try {
      await driveAsync(gen, { budgetMs: 0 });
    } finally {
      clearInterval(interval);
    }
    expect(ticks).toBeGreaterThan(0);
  });
});
