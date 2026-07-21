import { describe, expect, it } from "vitest";
import { divergingColor, divergingLegend, robustLimit } from "./theme";

/** OKLab lightness — the same measure used to validate the ramp. */
function oklabL(hex: string): number {
  const toLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const n = parseInt(hex.slice(1), 16);
  const r = toLinear(((n >> 16) & 255) / 255);
  const g = toLinear(((n >> 8) & 255) / 255);
  const b = toLinear((n & 255) / 255);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
}

describe("diverging ramp", () => {
  for (const mode of ["light", "dark"] as const) {
    describe(mode, () => {
      it("puts zero exactly on the neutral midpoint", () => {
        const mid = mode === "light" ? "#f0efec" : "#383835";
        expect(divergingColor(0, 1, mode)).toBe(mid);
      });

      it("maps opposite signs to different arms", () => {
        expect(divergingColor(-0.9, 1, mode)).not.toBe(divergingColor(0.9, 1, mode));
      });

      it("is symmetric about zero in step index", () => {
        // Equal magnitudes must sit at equal distance from the midpoint, so a
        // reader cannot mistake -0.5 for being weaker than +0.5.
        const legend = divergingLegend(mode);
        const negIndex = legend.indexOf(divergingColor(-0.5, 1, mode));
        const posIndex = legend.indexOf(divergingColor(0.5, 1, mode));
        const midIndex = legend.indexOf(mode === "light" ? "#f0efec" : "#383835");
        expect(midIndex - negIndex).toBe(posIndex - midIndex);
      });

      it("is monotonic in lightness across the whole legend", () => {
        // In light mode the midpoint is the LIGHTEST point and arms darken
        // outward; in dark mode it is the darkest and arms lighten outward.
        const legend = divergingLegend(mode);
        const mid = Math.floor(legend.length / 2);
        const left = legend.slice(0, mid + 1).map(oklabL);
        const right = legend.slice(mid).map(oklabL);
        const rising = (arr: number[]) => arr.every((v, i) => i === 0 || v > arr[i - 1]);
        const falling = (arr: number[]) => arr.every((v, i) => i === 0 || v < arr[i - 1]);

        if (mode === "light") {
          expect(rising(left)).toBe(true);
          expect(falling(right)).toBe(true);
        } else {
          expect(falling(left)).toBe(true);
          expect(rising(right)).toBe(true);
        }
      });

      it("separates the extreme from the midpoint", () => {
        // Regression: the dark blue arm once ended at a lightness identical to
        // the dark midpoint, so "very negative" rendered as "zero".
        const midHex = mode === "light" ? "#f0efec" : "#383835";
        const extreme = divergingColor(-1, 1, mode);
        expect(Math.abs(oklabL(extreme) - oklabL(midHex))).toBeGreaterThan(0.15);
      });

      it("clamps beyond the limit rather than wrapping", () => {
        expect(divergingColor(5, 1, mode)).toBe(divergingColor(1, 1, mode));
        expect(divergingColor(-5, 1, mode)).toBe(divergingColor(-1, 1, mode));
      });

      it("gives the neutral a BAND around zero, not a single point", () => {
        // With continuous data an exact zero never occurs. If only 0.0 were
        // neutral, every cell would carry a hue and "near zero" would stop
        // reading as "nothing" — the entire job of a diverging scale.
        const mid = mode === "light" ? "#f0efec" : "#383835";
        expect(divergingColor(0.001, 1, mode)).toBe(mid);
        expect(divergingColor(-0.001, 1, mode)).toBe(mid);
        expect(divergingColor(0.5, 1, mode)).not.toBe(mid);
      });
    });
  }

  it("returns the midpoint for a degenerate limit", () => {
    expect(divergingColor(1, 0, "light")).toBe("#f0efec");
    expect(divergingColor(NaN, 1, "light")).toBe("#f0efec");
  });

  it("orders the legend negative -> mid -> positive", () => {
    const legend = divergingLegend("light");
    expect(legend).toHaveLength(13);
    expect(legend[6]).toBe("#f0efec");
  });
});

describe("robustLimit", () => {
  it("ignores a single extreme outlier", () => {
    // Scaling to max() let one outlier compress everything else into the
    // innermost band — 86% of real embedding cells landed on two near-identical
    // steps, so the heatmap carried almost no information.
    const values = [...Array(99).fill(0.02), 5.0];
    expect(robustLimit(values, 0.98)).toBeLessThan(0.1);
    expect(Math.max(...values.map(Math.abs))).toBe(5.0);
  });

  it("is symmetric in sign", () => {
    const positive = robustLimit([0.1, 0.2, 0.3, 0.4], 0.98);
    const negative = robustLimit([-0.1, -0.2, -0.3, -0.4], 0.98);
    expect(positive).toBe(negative);
  });

  it("survives degenerate input", () => {
    expect(robustLimit([])).toBe(1);
    expect(robustLimit([0, 0, 0])).toBeGreaterThan(0);
    expect(robustLimit([NaN, Infinity, 0.5])).toBeGreaterThan(0);
  });
});
