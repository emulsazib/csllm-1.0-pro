import { describe, expect, it } from "vitest";
import {
  type AttentionBlock,
  attentionByKey,
  attentionByLayerKey,
  attentionMatrix,
  attentionRow,
  attentionVector,
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

describe("attentionVector", () => {
  it("picks a single head", () => {
    const block = makeBlock(2, 2, 3);
    expect(Array.from(attentionVector(block, 1, 0))).toEqual(
      Array.from(attentionRow(block, 1, 0)),
    );
  });

  it("averages the layer's heads for 'all'", () => {
    const block: AttentionBlock = {
      data: new Float32Array([1, 0, 0, 1]), // layer 0: head0=[1,0], head1=[0,1]
      layers: 1,
      heads: 2,
      keys: 2,
    };
    expect(Array.from(attentionVector(block, 0, "all"))).toEqual([0.5, 0.5]);
  });

  it("clamps a layer/head that outlived a deeper model", () => {
    // Panels keep layer/head in state across generations. Without clamping,
    // a stale index reads past the buffer and renders adjacent memory as
    // attention — plausible-looking and completely wrong.
    const block = makeBlock(2, 2, 3);
    expect(Array.from(attentionVector(block, 99, 0))).toEqual(
      Array.from(attentionRow(block, 1, 0)),
    );
    expect(Array.from(attentionVector(block, 0, 99))).toEqual(
      Array.from(attentionRow(block, 0, 1)),
    );
    expect(Array.from(attentionVector(block, -5, -5))).toEqual(
      Array.from(attentionRow(block, 0, 0)),
    );
  });

  it("never reads past the end of the buffer", () => {
    const block = makeBlock(2, 2, 3);
    expect(attentionVector(block, 99, 99)).toHaveLength(block.keys);
  });
});

describe("attentionMatrix", () => {
  /** One decode step: `keys` grows by one each token, because decoding is causal. */
  function step(keys: number, fill: number): { attention: AttentionBlock } {
    return {
      attention: { data: new Float32Array(keys).fill(fill), layers: 1, heads: 1, keys },
    };
  }

  it("stacks tokens into rows and reports the widest", () => {
    const { rows, keys } = attentionMatrix([step(2, 0.5), step(3, 0.25)], 0, 0);
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveLength(2);
    expect(rows[1]).toHaveLength(3);
    expect(keys).toBe(3); // the widest row, not the first
  });

  it("stays ragged rather than padding short rows", () => {
    // Padding would draw "did not attend" and "could not attend" identically —
    // the early token had no such key in its context at all.
    const { rows } = attentionMatrix([step(1, 1), step(4, 0.25)], 0, 0);
    expect(rows[0]).toHaveLength(1);
    expect(rows[0][0]).toBe(1);
  });

  it("skips tokens with no attention block", () => {
    const { rows, keys } = attentionMatrix([{}, step(2, 0.5), {}], 0, 0);
    expect(rows).toHaveLength(1);
    expect(keys).toBe(2);
  });

  it("returns an empty matrix when nothing has attention", () => {
    expect(attentionMatrix([{}, {}], 0, 0)).toEqual({ rows: [], keys: 0 });
  });

  it("honours the head selection per row", () => {
    const block: AttentionBlock = {
      data: new Float32Array([1, 0, 0, 1]),
      layers: 1,
      heads: 2,
      keys: 2,
    };
    expect(Array.from(attentionMatrix([{ attention: block }], 0, 1).rows[0])).toEqual([0, 1]);
  });
});
