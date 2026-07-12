/**
 * Headless-browser Web Worker test runner.
 *
 * Bundles tests/browser/worker-page.ts (main thread) and worker-entry.ts
 * (the packaged worker entry, served both at /worker-entry.js and at
 * /worker.js so the client's default factory resolves too), opens the page
 * in Playwright Chromium, and reports each case. WebGPU flags are enabled
 * so the webgpu-in-worker smoke test can execute where an adapter exists —
 * its backend is reported, not required.
 *
 * Exit codes: 0 all cases passed; 1 failures.
 */
import { createServer } from 'node:http';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { chromium } from 'playwright';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

const PAGE_HTML = `<!doctype html>
<html><head><meta charset="utf-8"><title>pca-web worker tests</title></head>
<body><pre id="status">running…</pre><script type="module" src="/worker-page.js"></script></body></html>`;

async function bundle(entry) {
  const result = await build({
    entryPoints: [join(ROOT, 'tests', 'browser', entry)],
    bundle: true,
    format: 'esm',
    target: 'es2022',
    write: false,
    sourcemap: 'inline',
  });
  return result.outputFiles[0].text;
}

function serve(pageJs, workerJs) {
  const server = createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/' || url.pathname === '/index.html') {
      res.writeHead(200, { 'content-type': 'text/html' });
      res.end(PAGE_HTML);
    } else if (url.pathname === '/worker-page.js') {
      res.writeHead(200, { 'content-type': 'text/javascript' });
      res.end(pageJs);
    } else if (url.pathname === '/worker-entry.js' || url.pathname === '/worker.js') {
      // /worker.js serves the default-factory resolution
      // (new URL('./worker.js', import.meta.url) from /worker-page.js).
      res.writeHead(200, { 'content-type': 'text/javascript' });
      res.end(workerJs);
    } else {
      res.writeHead(404);
      res.end('not found');
    }
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

async function runInBrowser(url, headless) {
  const args = ['--enable-unsafe-webgpu', '--enable-features=WebGPU'];
  if (process.env.CI) {
    // Software WebGPU adapter for GPU-less CI runners.
    args.push('--enable-unsafe-swiftshader');
  }
  const browser = await chromium.launch({ headless, channel: 'chromium', args });
  try {
    const page = await browser.newPage();
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        console.error(`  [page console.error] ${msg.text()}`);
      }
    });
    page.on('pageerror', (err) => console.error(`  [page error] ${err.message}`));
    await page.goto(url);
    await page.waitForFunction(() => window.__PCA_WORKER_RESULTS__ !== undefined, undefined, {
      timeout: 300_000,
    });
    return await page.evaluate(() => window.__PCA_WORKER_RESULTS__);
  } finally {
    await browser.close();
  }
}

const [pageJs, workerJs] = await Promise.all([bundle('worker-page.ts'), bundle('worker-entry.ts')]);
const server = await serve(pageJs, workerJs);
const { port } = server.address();
const results = await runInBrowser(`http://127.0.0.1:${port}/`, true);
server.close();

console.log('\n=== pca-web Web Worker test report ===');
for (const c of results.cases) {
  console.log(`  ${c.pass ? 'PASS' : 'FAIL'}  ${c.id}: ${c.detail}`);
}
console.log(
  `\nworker WebGPU smoke backend: ${results.workerGpuBackend ?? 'errored'}; failures: ${results.failures}`,
);

if (results.failures > 0) {
  console.error(`\nRESULT: ${results.failures} worker test case(s) FAILED.`);
  process.exit(1);
}
console.log('\nRESULT: all worker tests passed.');
process.exit(0);
