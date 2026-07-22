/** Palette values the canvas/Chart.js layers need in JS.
 *
 *  CSS owns the token definitions (styles.css); this reads them back so there is
 *  one source of truth, with literals only as a pre-mount fallback.
 */

export type Mode = "light" | "dark";

export function currentMode(): Mode {
  if (typeof document === "undefined") return "light";
  const stamped = document.documentElement.getAttribute("data-theme");
  if (stamped === "dark" || stamped === "light") return stamped;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function cssVar(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/** Chart ink/chrome, resolved at render time so a theme flip repaints correctly. */
export function chartTokens() {
  return {
    text: cssVar("--text-secondary", "#52514e"),
    muted: cssVar("--text-muted", "#898781"),
    grid: cssVar("--gridline", "#e1e0d9"),
    surface: cssVar("--surface-1", "#fcfcfb"),
    series1: cssVar("--series-1", "#2a78d6"),
    series2: cssVar("--series-2", "#eb6834"),
  };
}

/** Blue sequential ramp, 100→700 (see the design palette).
 *
 *  For magnitudes with no polarity — attention weights live in [0,1] and a
 *  diverging scale would invent a midpoint that means nothing.
 */
const SEQUENTIAL_BLUE = [
  "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
  "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
];

/**
 * Magnitude in [0,1] → colour.
 *
 * Dark mode reverses the ramp so "more" is always further from the surface, i.e.
 * more visible, in both themes.
 */
export function sequentialColor(t: number, mode: Mode = currentMode()): string {
  if (!Number.isFinite(t)) return SEQUENTIAL_BLUE[0];
  const clamped = Math.max(0, Math.min(1, t));
  const index = Math.min(
    SEQUENTIAL_BLUE.length - 1,
    Math.floor(clamped * SEQUENTIAL_BLUE.length),
  );
  return mode === "dark"
    ? SEQUENTIAL_BLUE[SEQUENTIAL_BLUE.length - 1 - index]
    : SEQUENTIAL_BLUE[index];
}

export function sequentialLegend(mode: Mode = currentMode()): string[] {
  return mode === "dark" ? [...SEQUENTIAL_BLUE].reverse() : [...SEQUENTIAL_BLUE];
}

/** Diverging ramp for signed values (embeddings are centred near zero, so the
 *  sign is meaningful — a sequential ramp would hide it).
 *
 *  blue ↔ red with a neutral gray midpoint, equal steps per arm. Each arm was
 *  checked for monotonic OKLab lightness; in dark mode the midpoint is the
 *  DARKEST point and the arms lighten outward, which is why the dark blue arm is
 *  stepped differently rather than flipped.
 */
const DIVERGING: Record<Mode, { negative: string[]; mid: string; positive: string[] }> = {
  light: {
    // index 0 = most negative
    negative: ["#104281", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
    mid: "#f0efec",
    positive: ["#fbd7d7", "#f3a8a8", "#ea7a79", "#e34948", "#bf2f2e", "#8d1f1e"],
  },
  dark: {
    negative: ["#b7d3f6", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"],
    mid: "#383835",
    positive: ["#6b2422", "#8d1f1e", "#bf2f2e", "#e34948", "#ea7a79", "#f3a8a8"],
  },
};

/**
 * Map a value in [-limit, +limit] to a diverging colour.
 * `limit` is the symmetric extent, so zero always lands on the neutral midpoint.
 */
export function divergingColor(value: number, limit: number, mode: Mode = currentMode()): string {
  const ramp = DIVERGING[mode];
  if (!Number.isFinite(value) || limit <= 0) return ramp.mid;

  const t = Math.max(-1, Math.min(1, value / limit));
  const magnitude = Math.abs(t);

  const arm = t < 0 ? ramp.negative : ramp.positive;
  // The neutral occupies a BAND around zero, not a single point. With continuous
  // data an exact zero never occurs, so a point-neutral means every cell gets a
  // hue and "near zero" stops reading as "nothing" — which is the entire job of
  // a diverging scale.
  const steps = arm.length + 1;
  const index = Math.min(steps - 1, Math.floor(magnitude * steps));
  if (index === 0) return ramp.mid;
  // Arms are stored extreme-first for the negative side and centre-first for the
  // positive side, so the two index in opposite directions.
  return t < 0 ? arm[arm.length - index] : arm[index - 1];
}

/**
 * A robust symmetric extent for a diverging scale: the given percentile of |v|.
 *
 * Using max() lets one outlier compress everything else into the innermost band —
 * with real embeddings that put 86% of cells on two nearly identical steps. The
 * trade is that values beyond the limit clip, so callers should say so.
 */
export function robustLimit(values: number[], percentile = 0.98): number {
  const magnitudes = values.map(Math.abs).filter(Number.isFinite).sort((a, b) => a - b);
  if (!magnitudes.length) return 1;
  const index = Math.min(magnitudes.length - 1, Math.floor(percentile * magnitudes.length));
  return magnitudes[index] || magnitudes[magnitudes.length - 1] || 1;
}

export function divergingLegend(mode: Mode = currentMode()): string[] {
  const ramp = DIVERGING[mode];
  return [...ramp.negative, ramp.mid, ...ramp.positive];
}
