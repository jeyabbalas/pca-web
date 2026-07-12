/** ECharts wrappers (via echarts/core for tree-shaking) and raw-canvas image tiles. */
import { BarChart, LineChart, ScatterChart } from 'echarts/charts';
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components';
import type { EChartsCoreOption, EChartsType } from 'echarts/core';
import * as echarts from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import type { Matrix } from 'pca-web';

echarts.use([
  ScatterChart,
  BarChart,
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  CanvasRenderer,
]);

/** tab10-ish categorical palette, one color per digit/cluster label. */
export const PALETTE = [
  '#4e79a7',
  '#f28e2b',
  '#59a14f',
  '#e15759',
  '#76b7b2',
  '#edc948',
  '#b07aa1',
  '#ff9da7',
  '#9c755f',
  '#bab0ac',
];

const allCharts: EChartsType[] = [];

function makeChart(el: HTMLElement, option: EChartsCoreOption): EChartsType {
  const chart = echarts.init(el);
  chart.setOption(option);
  allCharts.push(chart);
  return chart;
}

/**
 * Re-measures every chart's container. Charts created inside a
 * `hidden` panel initialize at zero size — call this right after
 * revealing the panel.
 */
export function resizeAllCharts(): void {
  for (const c of allCharts) {
    c.resize();
  }
}

let resizeHooked = false;
function hookResize(): void {
  if (!resizeHooked) {
    resizeHooked = true;
    window.addEventListener('resize', resizeAllCharts);
  }
}

// ---------------------------------------------------------------------
// Embedding scatter (live during fits, refined per snapshot)
// ---------------------------------------------------------------------

export interface ScatterView {
  /** Redraws PC1×PC2 of `scores`, colored by label; `outliers` get red rings. */
  update(scores: Matrix, labels: Uint8Array | null, outliers?: Set<number>): void;
  clear(): void;
}

export function createScatter(el: HTMLElement): ScatterView {
  hookResize();
  const chart = makeChart(el, {
    animation: false,
    grid: { left: 45, right: 15, top: 30, bottom: 35 },
    legend: { top: 0, type: 'scroll', itemWidth: 12, itemHeight: 8 },
    xAxis: { type: 'value', scale: true, name: 'PC1' },
    yAxis: { type: 'value', scale: true, name: 'PC2' },
  });
  return {
    update(scores, labels, outliers) {
      const n = scores.rows;
      const twoD = scores.cols >= 2;
      const byClass = new Map<number, number[][]>();
      const outlierPts: number[][] = [];
      for (let i = 0; i < n; i++) {
        const x = scores.get(i, 0);
        const y = twoD ? scores.get(i, 1) : 0;
        if (outliers?.has(i)) {
          outlierPts.push([x, y]);
        }
        const c = labels === null ? 0 : labels[i];
        let arr = byClass.get(c);
        if (arr === undefined) {
          arr = [];
          byClass.set(c, arr);
        }
        arr.push([x, y]);
      }
      const series: object[] = [...byClass.keys()]
        .sort((a, b) => a - b)
        .map((c) => ({
          name: `${c}`,
          type: 'scatter',
          data: byClass.get(c),
          symbolSize: 3.5,
          large: true,
          largeThreshold: 5000,
          silent: true,
          itemStyle: { color: PALETTE[c % PALETTE.length] },
        }));
      if (outlierPts.length > 0) {
        series.push({
          name: 'outliers',
          type: 'scatter',
          data: outlierPts,
          symbolSize: 9,
          silent: true,
          itemStyle: { color: 'rgba(0,0,0,0)', borderColor: '#c0392b', borderWidth: 1.5 },
        });
      }
      chart.setOption({ series } as EChartsCoreOption, { replaceMerge: ['series'] });
    },
    clear() {
      chart.setOption({ series: [] } as EChartsCoreOption, { replaceMerge: ['series'] });
    },
  };
}

// ---------------------------------------------------------------------
// Scree plot
// ---------------------------------------------------------------------

export interface ScreeView {
  update(evr: ArrayLike<number>): void;
}

export function createScree(el: HTMLElement): ScreeView {
  hookResize();
  const chart = makeChart(el, {
    animation: false,
    grid: { left: 45, right: 15, top: 30, bottom: 35 },
    tooltip: { trigger: 'axis' },
    legend: { top: 0 },
    xAxis: { type: 'category', name: 'component' },
    yAxis: { type: 'value', min: 0, max: 1 },
  });
  return {
    update(evr) {
      const k = evr.length;
      const cats = new Array<string>(k);
      const bars = new Array<number>(k);
      const cum = new Array<number>(k);
      let acc = 0;
      for (let i = 0; i < k; i++) {
        cats[i] = `${i + 1}`;
        bars[i] = evr[i];
        acc += evr[i];
        cum[i] = Math.min(1, acc);
      }
      chart.setOption({
        xAxis: { data: cats },
        series: [
          { name: 'ratio', type: 'bar', data: bars, itemStyle: { color: PALETTE[0] } },
          {
            name: 'cumulative',
            type: 'line',
            data: cum,
            symbolSize: 4,
            itemStyle: { color: PALETTE[1] },
          },
        ],
      } as EChartsCoreOption);
    },
  };
}

// ---------------------------------------------------------------------
// MSE-vs-k line
// ---------------------------------------------------------------------

export interface LineView {
  update(xs: number[], ys: number[]): void;
}

export function createMseLine(el: HTMLElement): LineView {
  hookResize();
  const chart = makeChart(el, {
    animation: false,
    grid: { left: 60, right: 15, top: 25, bottom: 35 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'value', name: 'k', minInterval: 1 },
    yAxis: { type: 'value', name: 'MSE' },
  });
  return {
    update(xs, ys) {
      chart.setOption({
        series: [
          {
            name: 'reconstruction MSE',
            type: 'line',
            data: xs.map((x, i) => [x, ys[i]]),
            symbolSize: 4,
            itemStyle: { color: PALETTE[3] },
          },
        ],
      } as EChartsCoreOption);
    },
  };
}

// ---------------------------------------------------------------------
// scoreSamples histogram
// ---------------------------------------------------------------------

export interface HistogramView {
  /** Bins `values`; bars whose bin lies below `threshold` are drawn red. */
  update(values: Float64Array, threshold: number): void;
}

export function createHistogram(el: HTMLElement): HistogramView {
  hookResize();
  const chart = makeChart(el, {
    animation: false,
    grid: { left: 50, right: 15, top: 15, bottom: 45 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', name: 'log-likelihood', nameLocation: 'middle', nameGap: 28 },
    yAxis: { type: 'value', name: 'count' },
  });
  return {
    update(values, threshold) {
      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      for (const v of values) {
        if (v < min) {
          min = v;
        }
        if (v > max) {
          max = v;
        }
      }
      const bins = 40;
      const width = (max - min) / bins || 1;
      const counts = new Array<number>(bins).fill(0);
      for (const v of values) {
        counts[Math.min(bins - 1, Math.floor((v - min) / width))]++;
      }
      const cats = new Array<string>(bins);
      const data = new Array<object>(bins);
      for (let b = 0; b < bins; b++) {
        const center = min + (b + 0.5) * width;
        cats[b] = center.toFixed(1);
        data[b] = {
          value: counts[b],
          itemStyle: { color: center < threshold ? '#c0392b' : PALETTE[0] },
        };
      }
      chart.setOption({
        xAxis: { data: cats },
        series: [{ name: 'samples', type: 'bar', data, barCategoryGap: '10%' }],
      } as EChartsCoreOption);
    },
  };
}

// ---------------------------------------------------------------------
// Raw-canvas image tiles (eigen-digits, reconstructions)
// ---------------------------------------------------------------------

export type TileMode = 'diverging' | 'ink';

export interface Tile {
  values: ArrayLike<number>;
  caption: string;
}

/**
 * Renders side×side images as pixelated canvas tiles. 'diverging' maps
 * sign-symmetric values blue–white–red (eigenvectors); 'ink' maps 0..16
 * to white→dark (digit pixels).
 */
export function renderTiles(
  container: HTMLElement,
  tiles: Tile[],
  side: number,
  mode: TileMode,
  scale = 6,
): void {
  container.textContent = '';
  const tmp = document.createElement('canvas');
  tmp.width = side;
  tmp.height = side;
  const tmpCtx = tmp.getContext('2d');
  if (tmpCtx === null) {
    return;
  }
  for (const tile of tiles) {
    const img = tmpCtx.createImageData(side, side);
    const v = tile.values;
    if (mode === 'diverging') {
      let amax = 1e-12;
      for (let i = 0; i < v.length; i++) {
        const a = Math.abs(v[i]);
        if (a > amax) {
          amax = a;
        }
      }
      for (let i = 0; i < v.length; i++) {
        const t = Math.max(-1, Math.min(1, v[i] / amax));
        const o = i * 4;
        // white at 0; +1 → warm red (178,24,43); −1 → cool blue (33,102,172)
        if (t >= 0) {
          img.data[o] = 255 - Math.round((255 - 178) * t);
          img.data[o + 1] = 255 - Math.round((255 - 24) * t);
          img.data[o + 2] = 255 - Math.round((255 - 43) * t);
        } else {
          img.data[o] = 255 - Math.round((255 - 33) * -t);
          img.data[o + 1] = 255 - Math.round((255 - 102) * -t);
          img.data[o + 2] = 255 - Math.round((255 - 172) * -t);
        }
        img.data[o + 3] = 255;
      }
    } else {
      for (let i = 0; i < v.length; i++) {
        const t = Math.max(0, Math.min(16, v[i])) / 16;
        const c = 255 - Math.round(225 * t);
        const o = i * 4;
        img.data[o] = c;
        img.data[o + 1] = c;
        img.data[o + 2] = c;
        img.data[o + 3] = 255;
      }
    }
    tmpCtx.putImageData(img, 0, 0);
    const canvas = document.createElement('canvas');
    canvas.width = side * scale;
    canvas.height = side * scale;
    const ctx = canvas.getContext('2d');
    if (ctx !== null) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    }
    const fig = document.createElement('figure');
    fig.appendChild(canvas);
    const cap = document.createElement('figcaption');
    cap.textContent = tile.caption;
    fig.appendChild(cap);
    container.appendChild(fig);
  }
}
