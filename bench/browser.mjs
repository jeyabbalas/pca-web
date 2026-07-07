/**
 * CPU vs WebGPU benchmark runner: bundles bench/browser-page.ts, opens it in
 * Chromium with WebGPU enabled, and prints the timing table.
 * Run: npm run bench:browser
 */
import { createServer } from 'node:http';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { chromium } from 'playwright';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

const HTML = `<!doctype html>
<html><head><meta charset="utf-8"><title>pca-web bench</title></head>
<body><pre>benchmarking…</pre><script type="module" src="/page.js"></script></body></html>`;

const { outputFiles } = await build({
  entryPoints: [join(ROOT, 'bench', 'browser-page.ts')],
  bundle: true,
  format: 'esm',
  target: 'es2022',
  write: false,
});
const pageJs = outputFiles[0].text;

const server = createServer((req, res) => {
  if (req.url === '/page.js') {
    res.writeHead(200, { 'content-type': 'text/javascript' });
    res.end(pageJs);
  } else {
    res.writeHead(200, { 'content-type': 'text/html' });
    res.end(HTML);
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));

const browser = await chromium.launch({
  headless: true,
  channel: 'chromium',
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU'],
});
const page = await browser.newPage();
page.on('pageerror', (err) => console.error(`[page error] ${err.message}`));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
await page.waitForFunction(() => window.__PCA_BENCH__ !== undefined, undefined, {
  timeout: 600_000,
});
const results = await page.evaluate(() => window.__PCA_BENCH__);
await browser.close();
server.close();

if (results.error) {
  console.error(`Benchmark failed: ${results.error}`);
  process.exit(1);
}
console.log(`\npca-web CPU vs WebGPU fit benchmarks (adapter: ${results.adapter ?? 'none'})\n`);
console.log(
  'case                     size          solver            cpu(ms)   gpu(ms)  speedup  backend  maxRelDiff',
);
for (const r of results.rows) {
  console.log(
    `${r.id.padEnd(24)} ${`${r.n}x${r.p}`.padEnd(13)} ${r.solver.padEnd(17)} ${r.cpuMs
      .toFixed(0)
      .padStart(8)} ${r.gpuMs.toFixed(0).padStart(9)} ${(r.cpuMs / r.gpuMs)
      .toFixed(2)
      .padStart(8)}x ${r.backend.padEnd(8)} ${r.maxRelDiff.toExponential(1)}`,
  );
}
