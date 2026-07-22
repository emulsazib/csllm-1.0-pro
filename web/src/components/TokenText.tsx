/** Rendering token text so whitespace is visible.
 *
 *  A token's text is frequently " the" or "\n\n" — printed raw, a chip carrying a
 *  leading space is indistinguishable from one without, which is exactly the
 *  distinction byte-level BPE turns on. Shared by the tokenizer view and the
 *  playground so the two cannot diverge on what a token "looks like".
 *
 *  This is the MARKUP form, which can style each glyph separately. For plain
 *  strings (table cells, chart axes) use `tokenLabel` from ProbabilityChart.
 */

const GLYPHS: Record<string, string> = {
  "\n": "↵",
  "\t": "⇥",
  " ": "␣",
};

/** Alternating tints keep adjacent tokens separable without implying categories.
 *  Deliberately NOT the categorical series colours — token index carries no
 *  identity, so using series hues would suggest a grouping that does not exist. */
export const TOKEN_TINTS = ["rgba(42,120,214,0.13)", "rgba(42,120,214,0.05)"];

/** Split on whitespace runs, keeping the separators so each gets its own glyph. */
export function TokenText({ text }: { text: string }) {
  if (text === "") return <span className="ws">∅</span>;
  const parts = text.split(/(\n|\t| )/g).filter((p) => p !== "");
  return (
    <>
      {parts.map((part, i) =>
        GLYPHS[part] ? (
          <span key={i} className="ws">
            {GLYPHS[part]}
          </span>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}
