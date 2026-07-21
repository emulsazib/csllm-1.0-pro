/** Training and validation loss over steps.
 *
 *  Two series on ONE axis — both are loss in the same units, so they are directly
 *  comparable and the gap between them IS the overfitting signal. Learning rate
 *  and gradient norm live in their own charts rather than on a second y-axis:
 *  a dual-axis chart lets the reader infer a relationship from an arbitrary
 *  scale choice.
 */

import {
  CategoryScale,
  Chart as ChartJS,
  Decimation,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
  type TooltipItem,
} from "chart.js";
import { Line } from "react-chartjs-2";
import type { EvalPoint, LossPoint } from "../hooks/useTrainingStream";
import { chartTokens, currentMode } from "../theme";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
  Decimation,
);

export function LossChart({ train, val }: { train: LossPoint[]; val: EvalPoint[] }) {
  const tokens = chartTokens();
  const mode = currentMode();

  const data = {
    datasets: [
      {
        label: "Train",
        data: train.map((p) => ({ x: p.step, y: p.loss })),
        borderColor: tokens.series1,
        backgroundColor: tokens.series1,
        borderWidth: 2,
        pointRadius: 0,
        // Markers appear on hover only; a dot on every step would be noise at
        // thousands of points.
        pointHoverRadius: 4,
        tension: 0.15,
      },
      {
        label: "Validation",
        data: val.map((p) => ({ x: p.step, y: p.val })),
        borderColor: tokens.series2,
        backgroundColor: tokens.series2,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 6,
        tension: 0,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false as const,
    parsing: false as const,
    interaction: { mode: "nearest" as const, axis: "x" as const, intersect: false },
    plugins: {
      // Long runs emit one point per step; decimation keeps the draw cheap
      // without changing the shape of the curve.
      decimation: { enabled: true, algorithm: "lttb" as const, samples: 400 },
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
        callbacks: {
          title: (items: TooltipItem<"line">[]) => `step ${items[0].parsed.x}`,
          label: (item: TooltipItem<"line">) =>
            `${item.dataset.label}: ${(item.parsed.y ?? 0).toFixed(4)}`,
        },
      },
    },
    scales: {
      x: {
        type: "linear" as const,
        border: { display: false },
        grid: { color: tokens.grid, drawTicks: false },
        ticks: { color: tokens.muted, font: { size: 11 }, maxTicksLimit: 8 },
        title: { display: true, text: "step", color: tokens.muted, font: { size: 11 } },
      },
      y: {
        border: { display: false },
        grid: { color: tokens.grid, drawTicks: false },
        ticks: { color: tokens.muted, font: { size: 11 } },
        title: { display: true, text: "loss (nats)", color: tokens.muted, font: { size: 11 } },
      },
    },
  };

  return (
    <div style={{ height: 260 }}>
      <Line key={mode} data={data} options={options} />
    </div>
  );
}

/** A single-series sparkline — no legend, because the title names the series. */
export function Sparkline({
  points,
  label,
  color,
}: {
  points: LossPoint[];
  label: string;
  color?: string;
}) {
  const tokens = chartTokens();
  const mode = currentMode();
  const stroke = color ?? tokens.series1;

  const data = {
    datasets: [
      {
        label,
        data: points.map((p) => ({ x: p.step, y: p.loss })),
        borderColor: stroke,
        backgroundColor: stroke,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.2,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false as const,
    parsing: false as const,
    plugins: {
      legend: { display: false },
      decimation: { enabled: true, algorithm: "lttb" as const, samples: 200 },
      tooltip: {
        backgroundColor: mode === "dark" ? "#2c2c2a" : "#0b0b0b",
        titleColor: "#ffffff",
        bodyColor: "#ffffff",
        padding: 8,
        cornerRadius: 6,
        callbacks: {
          title: (items: TooltipItem<"line">[]) => `step ${items[0].parsed.x}`,
          label: (item: TooltipItem<"line">) =>
          `${label}: ${(item.parsed.y ?? 0).toPrecision(4)}`,
        },
      },
    },
    scales: {
      x: { type: "linear" as const, display: false },
      y: {
        border: { display: false },
        grid: { color: tokens.grid, drawTicks: false },
        ticks: { color: tokens.muted, font: { size: 10 }, maxTicksLimit: 3 },
      },
    },
  };

  return (
    <div style={{ height: 110 }}>
      <Line key={mode} data={data} options={options} />
    </div>
  );
}
