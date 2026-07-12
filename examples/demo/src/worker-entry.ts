// Runs inside the module worker. The side-effect import attaches the pca-web
// request handler to the worker's global scope; WorkerPCA / WorkerIncrementalPCA
// on the main thread talk to it over postMessage.
import 'pca-web/worker';
