/** Pure logic behind the configurator.
 *
 *  `validHeadCounts` is the one place the UI encodes an engine invariant, so it
 *  is the one place worth testing: offering a head count the engine rejects
 *  turns a slider into a 422.
 */

import { describe, expect, it } from "vitest";
import debugConfig from "../../../configs/debug.json";
import shakespeareConfig from "../../../configs/shakespeare.json";
import { formatBytes, formatParams, PRESETS, validHeadCounts } from "./ConfiguratorPanel";

describe("presets track the shipped configs", () => {
  // A preset that drifts from configs/*.json quotes a parameter count for a
  // model that no config file can produce. ffn_hidden already drifted once.
  it.each([
    ["debug", debugConfig],
    ["shakespeare", shakespeareConfig],
  ])("%s matches configs/%s.json", (name, file) => {
    const { _comment, ...shipped } = file as Record<string, unknown>;
    expect(PRESETS[name]).toEqual(shipped);
  });
});

describe("validHeadCounts", () => {
  it("only offers divisors that leave an even head_dim", () => {
    for (const nEmbd of [32, 64, 96, 384, 512, 768]) {
      for (const h of validHeadCounts(nEmbd)) {
        expect(nEmbd % h).toBe(0);
        expect((nEmbd / h) % 2).toBe(0); // RoPE rotates channel pairs
      }
    }
  });

  it("excludes divisors that would make head_dim odd", () => {
    // 384/128 = 3 — a clean divisor, but an odd head_dim the engine rejects.
    expect(384 % 128).toBe(0);
    expect(validHeadCounts(384)).not.toContain(128);
  });

  it("covers the shipped configs", () => {
    expect(validHeadCounts(384)).toContain(6); // configs/shakespeare.json
    expect(validHeadCounts(64)).toContain(2); // configs/debug.json
  });

  it("respects the API's 64-head upper bound", () => {
    expect(Math.max(...validHeadCounts(2048))).toBeLessThanOrEqual(64);
  });

  it("always offers at least one option for even widths", () => {
    for (let nEmbd = 32; nEmbd <= 2048; nEmbd += 32) {
      expect(validHeadCounts(nEmbd).length).toBeGreaterThan(0);
    }
  });
});

describe("formatParams", () => {
  it("renders the documented 12.19 M", () => {
    expect(formatParams(12_194_688)).toBe("12.19 M");
  });

  it("scales across magnitudes", () => {
    expect(formatParams(139_584)).toBe("139.6 K");
    expect(formatParams(1_500_000_000)).toBe("1.50 B");
    expect(formatParams(842)).toBe("842");
  });
});

describe("formatBytes", () => {
  it("uses binary units", () => {
    expect(formatBytes(1024)).toBe("1.0 KB");
    expect(formatBytes(1024 ** 2)).toBe("1.0 MB");
    expect(formatBytes(1024 ** 3)).toBe("1.00 GB");
    expect(formatBytes(512)).toBe("512 B");
  });
});
