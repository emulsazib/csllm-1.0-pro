import { useEffect, useState } from "react";
import { api } from "./api/client";
import type { Health } from "./api/types";
import { AttentionPanel } from "./components/AttentionPanel";
import { ConfiguratorPanel } from "./components/ConfiguratorPanel";
import { DatasetBrowser } from "./components/DatasetBrowser";
import { ExportModal } from "./components/ExportModal";
import { PlaygroundPanel } from "./components/PlaygroundPanel";
import { ProbabilityPanel } from "./components/ProbabilityPanel";
import { TokenizerPanel } from "./components/TokenizerPanel";
import { TrainingDashboard } from "./components/TrainingDashboard";
import { currentMode, type Mode } from "./theme";

type TabId =
  | "configure"
  | "datasets"
  | "playground"
  | "tokens"
  | "sampling"
  | "attention"
  | "training";

const TABS: { id: TabId; label: string; ready: boolean }[] = [
  { id: "configure", label: "Configure", ready: true },
  { id: "datasets", label: "Datasets", ready: true },
  { id: "playground", label: "Playground", ready: true },
  { id: "tokens", label: "Tokens & Embeddings", ready: true },
  { id: "sampling", label: "Sampling", ready: true },
  { id: "attention", label: "Attention", ready: true },
  { id: "training", label: "Training", ready: true },
];

function ThemeToggle() {
  const [mode, setMode] = useState<Mode>(currentMode);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", mode);
  }, [mode]);

  return (
    <button
      className="ghost"
      onClick={() => setMode(mode === "dark" ? "light" : "dark")}
      aria-label={`Switch to ${mode === "dark" ? "light" : "dark"} theme`}
    >
      {mode === "dark" ? "Light" : "Dark"}
    </button>
  );
}

/** Tabs live in the URL hash so a view is linkable and survives a reload. */
function initialTab(): TabId {
  const hash = typeof window === "undefined" ? "" : window.location.hash.replace("#", "");
  const match = TABS.find((t) => t.id === hash && t.ready);
  return match ? match.id : "tokens";
}

export default function App() {
  const [tab, setTab] = useState<TabId>(initialTab);

  useEffect(() => {
    const onHashChange = () => setTab(initialTab());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  function selectTab(id: TabId) {
    setTab(id);
    window.location.hash = id;
  }

  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  // In the header rather than a tab: exporting is an action on the model, not a
  // view of it, and it is wanted from wherever you happen to be.
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    api
      .health(controller.signal)
      .then(setHealth)
      .catch((err: unknown) => {
        if (!controller.signal.aborted) {
          setHealthError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => controller.abort();
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>CSLLM Diagnostics</h1>
        {health && (
          <span className="meta">
            {health.num_params.toLocaleString()} params · {health.n_layer}L × {health.n_head}H ×{" "}
            {health.n_embd}d · ctx {health.block_size} · vocab{" "}
            {health.vocab_size.toLocaleString()} · {health.blas_backend}
          </span>
        )}
        <span className="spacer" />
        <button className="ghost" onClick={() => setExporting(true)}>
          Export
        </button>
        <ThemeToggle />
      </header>

      {exporting && <ExportModal onClose={() => setExporting(false)} />}

      {healthError && (
        <div className="error">
          Cannot reach the gateway: {healthError}. Start it with <code>make serve</code>.
        </div>
      )}

      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            className="tab"
            role="tab"
            aria-selected={tab === t.id}
            disabled={!t.ready}
            title={t.ready ? undefined : "Arrives in the next phase"}
            onClick={() => selectTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {tab === "configure" && <ConfiguratorPanel />}
      {tab === "datasets" && <DatasetBrowser />}
      {tab === "playground" && <PlaygroundPanel />}
      {tab === "tokens" && <TokenizerPanel />}
      {tab === "sampling" && <ProbabilityPanel />}
      {tab === "attention" && <AttentionPanel />}
      {tab === "training" && <TrainingDashboard />}
    </div>
  );
}
