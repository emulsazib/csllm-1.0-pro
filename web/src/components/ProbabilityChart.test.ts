import { describe, expect, it } from "vitest";
import { tokenLabel } from "./ProbabilityChart";

describe("tokenLabel", () => {
  it("makes whitespace visible", () => {
    // A bar labelled with a bare space would look like a blank row.
    expect(tokenLabel(" the")).toBe("␣the");
    expect(tokenLabel("\n")).toBe("\\n");
    expect(tokenLabel("\t")).toBe("\\t");
    expect(tokenLabel("\r")).toBe("\\r");
  });

  it("marks the empty string", () => {
    expect(tokenLabel("")).toBe("∅");
  });

  it("leaves ordinary text alone", () => {
    expect(tokenLabel("KING")).toBe("KING");
  });

  it("handles multi-byte characters", () => {
    expect(tokenLabel("🌍")).toBe("🌍");
    expect(tokenLabel(" 🦀 ")).toBe("␣🦀␣");
  });
});
