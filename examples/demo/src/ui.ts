/** Small typed DOM helpers: element lookup, status lines, progress bar, liveness widget. */

export function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (el === null) {
    throw new Error(`missing element #${id}`);
  }
  return el as T;
}

export function inputEl(id: string): HTMLInputElement {
  return byId<HTMLInputElement>(id);
}

export function readRadio(name: string): string {
  const el = document.querySelector<HTMLInputElement>(`input[name="${name}"]:checked`);
  return el?.value ?? '';
}

export function onRadioChange(name: string, handler: (value: string) => void): void {
  for (const el of document.querySelectorAll<HTMLInputElement>(`input[name="${name}"]`)) {
    el.addEventListener('change', () => handler(el.value));
  }
}

export function setRadioDisabled(name: string, disabled: boolean): void {
  for (const el of document.querySelectorAll<HTMLInputElement>(`input[name="${name}"]`)) {
    el.disabled = disabled;
  }
}

export function setStatus(id: string, text: string, kind: 'ok' | 'err' | '' = ''): void {
  const el = byId(id);
  el.textContent = text;
  el.className = `status${kind === '' ? '' : ` ${kind}`}`;
}

/** Mirrors a range slider's value into its <output> sibling, now and on input. */
export function bindSliderOutput(sliderId: string, outputId: string): void {
  const slider = inputEl(sliderId);
  const out = byId<HTMLOutputElement>(outputId);
  const sync = () => {
    out.textContent = slider.value;
  };
  slider.addEventListener('input', sync);
  sync();
}

export function setProgress(fraction: number | null, label: string): void {
  const fill = byId('progress-fill');
  if (fraction === null) {
    fill.classList.add('indeterminate');
  } else {
    fill.classList.remove('indeterminate');
    fill.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  }
  byId('progress-label').textContent = label;
}

export function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(1)} ms`;
}

export function download(filename: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** Prepends a row to the run-history table, keeping the newest 10. */
export function addRunRow(cells: string[]): void {
  const body = byId<HTMLTableSectionElement>('history-body');
  const tr = document.createElement('tr');
  for (const c of cells) {
    const td = document.createElement('td');
    td.textContent = c;
    tr.appendChild(td);
  }
  body.insertBefore(tr, body.firstChild);
  while (body.children.length > 10) {
    body.removeChild(body.lastChild as Node);
  }
}

/**
 * Main-thread liveness widget: a requestAnimationFrame loop rotates the
 * spinner and reports FPS. Deliberately JS-driven — it visibly freezes
 * during a blocking fit, which is the point of the comparison.
 */
export function startLiveness(): void {
  const spinner = byId('spinner');
  const fps = byId('fps');
  let frames = 0;
  let windowStart = performance.now();
  let angle = 0;
  let last = windowStart;
  const tick = (now: number) => {
    angle += (now - last) * 0.18;
    last = now;
    spinner.style.transform = `rotate(${angle % 360}deg)`;
    frames++;
    if (now - windowStart >= 500) {
      fps.textContent = `${Math.round((frames * 1000) / (now - windowStart))} fps`;
      frames = 0;
      windowStart = now;
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
