/** Next-token probabilities, before and after the sampling filters.
 *
 *  Form: horizontal bars — the categories are token strings, which read far
 *  better along the y-axis than rotated under a vertical axis.
 *
 *  Two series, so a legend is always present:
 *    · blue   "after filters" — what the sampler would actually draw from
 *    · orange "model belief"  — the unfiltered softmax, independent of the knobs
 *
 *  An orange bar with no blue beside it is the whole point of the chart: the
 *  model wanted that token and your settings excluded it.
 */

import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Tooltip,
  type TooltipItem,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import type { CandidateToken } from "../api/types";
import { chartTokens, currentMode } from "../theme";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

/** Whitespace and control characters need to be visible as labels. */
export function tokenLabel(text: string): string {
  if (text === "") return "∅";
  return text
    .replace(/\n/g, "\\n")
    .replace(/\t/g, "\\t")
    .replace(/\r/g, "\\r")
    .replace(/ /g, "␣");
}

export function ProbabilityChart({ candidates }: { candidates: CandidateToken[] }) {
  const tokens = chartTokens();
  const mode = currentMode();

  const data = {
    labels: candidates.map((c) => tokenLabel(c.text)),
    datasets: [
      {
        label: "After filters",
        data: candidates.map((c) => c.prob),
        backgroundColor: tokens.series1,
        // 4px rounded data-end, anchored to the baseline (the flat end).
        borderRadius: 4,
        borderSkipped: "start" as const,
        // 2px of surface between adjacent bars.
        barPercentage: 0.82,
        categoryPercentage: 0.86,
      },
      {
        label: "Model belief",
        data: candidates.map((c) => c.raw_prob),
        backgroundColor: tokens.series2,
        borderRadius: 4,
        borderSkipped: "start" as const,
        barPercentage: 0.82,
        categoryPercentage: 0.86,
      },
    ],
  };

  const options = {
    indexAxis: "y" as const,
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 150 },
    layout: { padding: { right: 8 } },
    plugins: {
      legend: {
        position: "top" as const,
        align: "start" as const,
        labels: {
          color: tokens.text,
          boxWidth: 10,
          boxHeight: 10,
          borderRadius: 3,
          useBorderRadius: true,
          font: { size: 12 },
        },
      },
      tooltip: {
        backgroundColor: mode === "dark" ? "#2c2c2a" : "#0b0b0b",
        titleColor: "#ffffff",
        bodyColor: "#ffffff",
        padding: 10,
        cornerRadius: 6,
        displayColors: true,
        callbacks: {
          label: (item: TooltipItem<"bar">) => {
            const candidate = candidates[item.dataIndex];
            const value = item.parsed.x ?? 0;
            const pct = `${(value * 100).toFixed(2)}%`;
            if (item.datasetIndex === 0 && !candidate.kept) {
              return "After filters: excluded";
            }
            return `${item.dataset.label}: ${pct}`;
          },
          afterBody: (items: TooltipItem<"bar">[]) => {
            const candidate = candidates[items[0].dataIndex];
            return [`id ${candidate.id}  ·  logit ${candidate.logit.toFixed(3)}`];
          },
        },
      },
    },
    scales: {
      x: {
        beginAtZero: true,
        border: { display: false },
        grid: { color: tokens.grid, drawTicks: false },
        ticks: {
          color: tokens.muted,
          font: { size: 11 },
          callback: (value: string | number) => `${(Number(value) * 100).toFixed(0)}%`,
        },
      },
      y: {
        border: { color: tokens.grid },
        // No horizontal gridlines — the bars already carry the categories.
        grid: { display: false },
        ticks: { color: tokens.text, font: { size: 12, family: "ui-monospace, monospace" } },
      },
    },
  };

  return (
    // Keyed on mode so a theme flip rebuilds the chart with the other mode's
    // steps — Chart.js resolves colours once at config time.
    <div style={{ height: Math.max(220, candidates.length * 26 + 60) }}>
      <Bar key={mode} data={data} options={options} />
    </div>
  );
}
