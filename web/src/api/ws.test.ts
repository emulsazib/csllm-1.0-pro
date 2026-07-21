import { describe, expect, it } from "vitest";
import {
  type AttentionBlock,
  attentionByKey,
  attentionByLayerKey,
  attentionRow,
  parseAttention,
} from "./ws";

/** Build a block whose value encodes its own [layer, head, key] index. */
function makeBlock(layers: number, heads: number, keys: number): AttentionBlock {
  const data = new Float32Array(layers * heads * keys);
  let i = 0;
  for (let l = 0; l < layers; l++) {
    for (let h = 0; h < heads; h++) {
      for (let k = 0; k < keys; k++) data[i++] = l * 100 + h * 10 + k;
    }
  }
  return { data, layers, heads, keys };
}

describe("parseAttention", () => {
  it("reads a float32 frame into the declared shape", () => {
    const source = new Float32Array([0.1, 0.9, 0.5, 0.5, 0.25, 0.75]);
    const block = parseAttention([3, 1, 2], source.buffer);
    expect(block.layers).toBe(3);
    expect(block.heads).toBe(1);
    expect(block.keys).toBe(2);
    expect(Array.from(block.data)).toEqual(Array.from(source));
  });

  it("rejects a frame whose length disagrees with the shape", () => {
    // A silent mismatch would render one token's attention under another's
    // label — worse than an error, because it still looks plausible.
    const buffer = new Float32Array(5).buffer;
    expect(() => parseAttention([2, 2, 2], buffer)).toThrow(/expected 32/);
  });

  it("accepts an exactly-sized empty frame", () => {
    expect(() => parseAttention([0, 0, 0], new ArrayBuffer(0))).not.toThrow();
  });
});

describe("attentionRow", () => {
  it("indexes [layer][head] in row-major order", () => {
    const block = makeBlock(3, 2, 4);
    expect(Array.from(attentionRow(block, 0, 0))).toEqual([0, 1, 2, 3]);
    expect(Array.from(attentionRow(block, 0, 1))).toEqual([10, 11, 12, 13]);
    expect(Array.from(attentionRow(block, 2, 1))).toEqual([210, 211, 212, 213]);
  });

  it("returns a view, not a copy", () => {
    const block = makeBlock(1, 1, 3);
    attentionRow(block, 0, 0)[0] = 99;
    expect(block.data[0]).toBe(99);
  });
});

describe("aggregations", () => {
  it("averages every layer and head per key", () => {
    // Two heads over one layer: [1,0] and [0,1] -> mean [0.5, 0.5].
    const block: AttentionBlock = {
      data: new Float32Array([1, 0, 0, 1]),
      layers: 1,
      heads: 2,
      keys: 2,
    };
    expect(Array.from(attentionByKey(block))).toEqual([0.5, 0.5]);
  });

  it("averages heads within a single layer", () => {
    const block: AttentionBlock = {
      data: new Float32Array([
        1, 0, 0, 1, // layer 0: two heads
        0.5, 0.5, 0.5, 0.5, // layer 1
      ]),
      layers: 2,
      heads: 2,
      keys: 2,
    };
    expect(Array.from(attentionByLayerKey(block, 0))).toEqual([0.5, 0.5]);
    expect(Array.from(attentionByLayerKey(block, 1))).toEqual([0.5, 0.5]);
  });

  it("preserves a distribution when averaging rows that each sum to 1", () => {
    const block: AttentionBlock = {
      data: new Float32Array([0.7, 0.3, 0.2, 0.8]),
      layers: 1,
      heads: 2,
      keys: 2,
    };
    const mean = attentionByKey(block);
    expect(mean[0] + mean[1]).toBeCloseTo(1, 6);
  });

  it("does not divide by zero on an empty block", () => {
    const block: AttentionBlock = {
      data: new Float32Array(0),
      layers: 0,
      heads: 0,
      keys: 0,
    };
    expect(Array.from(attentionByKey(block))).toEqual([]);
  });
});
