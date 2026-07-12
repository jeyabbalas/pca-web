/**
 * Headless-browser GPU test runner.
 *
 * Bundles tests/browser/page.ts with esbuild, serves it (plus /fixtures)
 * over a loopback HTTP server, opens it in Playwright Chromium with WebGPU
 * enabled, and reports EXACTLY which GPU tests executed, on which adapter,
 * with what measured diffs.
 *
 * Exit codes: 0 all executed GPU tests passed; 1 test failures;
 * 2 the GPU path could not be executed in this environment at all.
 */
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { chromium } from 'playwright';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

const PAGE_HTML = `<!doctype html>
<html><head><meta charset="utf-8"><title>pca-web GPU tests</title></head>
<body><pre id="status">running…</pre><script type="module" src="/page.js"></script></body></html>`;

async function bundle() {
  const result = await build({
    entryPoints: [join(ROOT, 'tests', 'browser', 'page.ts')],
    bundle: true,
    format: 'esm',
    target: 'es2022',
    write: false,
    sourcemap: 'inline',
  });
  return result.outputFiles[0].text;
}

function serve(pageJs) {
  const types = { '.json': 'application/json', '.bin': 'application/octet-stream' };
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://localhost');
      if (url.pathname === '/' || url.pathname === '/index.html') {
        res.writeHead(200, { 'content-type': 'text/html' });
        res.end(PAGE_HTML);
      } else if (url.pathname === '/page.js') {
        res.writeHead(200, { 'content-type': 'text/javascript' });
        res.end(pageJs);
      } else if (url.pathname.startsWith('/fixtures/')) {
        const file = join(ROOT, url.pathname.slice(1));
        const body = await readFile(file);
        res.writeHead(200, {
          'content-type': types[extname(file)] ?? 'application/octet-stream',
        });
        res.end(body);
      } else {
        res.writeHead(404);
        res.end('not found');
      }
    } catch (err) {
      res.writeHead(500);
      res.end(String(err));
    }
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

async function runInBrowser(url, headless) {
  // channel 'chromium' selects the full Chromium build, whose "new headless"
  // mode supports WebGPU (the default headless shell does not).
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
    await page.waitForFunction(() => window.__PCA_RESULTS__ !== undefined, undefined, {
      timeout: 300_000,
    });
    return await page.evaluate(() => window.__PCA_RESULTS__);
  } finally {
    await browser.close();
  }
}

function report(results, mode) {
  const { adapter, dsRelError, cases, failures, gpuExecuted } = results;
  console.log(`\n=== pca-web GPU test report (${mode}) ===`);
  console.log(`WebGPU supported: ${results.supported}`);
  console.log(`Adapter:          ${adapter ?? 'none'}`);
  if (dsRelError !== null) {
    console.log(`df64 self-check:  relErr=${Number(dsRelError).toExponential(2)}`);
  }
  const bySection = { engine: [], equiv: [], fixtures: [] };
  for (const c of cases) {
    bySection[c.section]?.push(c);
  }
  for (const [section, list] of Object.entries(bySection)) {
    if (list.length === 0) {
      continue;
    }
    const passed = list.filter((c) => c.pass).length;
    console.log(`\n-- ${section} (${passed}/${list.length} passed) --`);
    for (const c of list) {
      console.log(`  ${c.pass ? 'PASS' : 'FAIL'}  [${c.backend}] ${c.id}: ${c.detail}`);
    }
  }
  const gpuCount = cases.filter((c) => c.backend === 'webgpu').length;
  console.log(
    `\nGPU-executed test cases: ${gpuCount}/${cases.length}; failures: ${failures}; gpuExecuted=${gpuExecuted}`,
  );
}

const pageJs = await bundle();
const server = await serve(pageJs);
const { port } = server.address();
const url = `http://127.0.0.1:${port}/`;

/** A headed retry can only work where a display server exists. */
function canRunHeaded() {
  return (
    process.platform !== 'linux' || Boolean(process.env.DISPLAY ?? process.env.WAYLAND_DISPLAY)
  );
}

let mode = 'headless';
let results = await runInBrowser(url, true);
if (!results.gpuExecuted && canRunHeaded()) {
  console.log('Headless run did not reach the GPU; retrying headed…');
  try {
    results = await runInBrowser(url, false);
    mode = 'headed';
  } catch (err) {
    // A display-less environment kills Chromium at launch; keep the headless
    // report and fall through to the exit-2 "not executable" path.
    console.error(`  headed retry failed to launch: ${String(err).split('\n')[0]}`);
  }
}
report(results, mode);
server.close();

if (!results.gpuExecuted) {
  console.error(
    '\nRESULT: the GPU path COULD NOT BE EXECUTED in this environment (no usable WebGPU adapter in headless or headed Chromium).',
  );
  process.exit(2);
}
if (results.failures > 0) {
  console.error(`\nRESULT: ${results.failures} GPU test case(s) FAILED.`);
  process.exit(1);
}
console.log('\nRESULT: all executed GPU tests passed.');
process.exit(0);
