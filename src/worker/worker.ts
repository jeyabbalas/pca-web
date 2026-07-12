/**
 * `pca-web/worker` — drop-in worker entry.
 *
 * Spawned directly (`new Worker(new URL('pca-web/worker', import.meta.url),
 * { type: 'module' })` or via a one-line `import 'pca-web/worker'` in your
 * own worker file), it attaches the pca-web protocol to the worker scope.
 * Importing it from a non-worker context is harmless: auto-attach happens
 * only inside a real DedicatedWorkerGlobalScope, so this module doubles as
 * the programmatic entry — `attachPCAWorker(port)` works with any message
 * port, including Node's `worker_threads` `parentPort`.
 */

import { attachPCAWorker } from './handler.js';
import type { PCAWorkerPort } from './protocol.js';

export { type AttachPCAWorkerOptions, attachPCAWorker } from './handler.js';
export * from './protocol.js';

const scope = globalThis as { DedicatedWorkerGlobalScope?: abstract new () => unknown };
if (
  typeof scope.DedicatedWorkerGlobalScope === 'function' &&
  globalThis instanceof scope.DedicatedWorkerGlobalScope
) {
  attachPCAWorker(globalThis as unknown as PCAWorkerPort);
}
