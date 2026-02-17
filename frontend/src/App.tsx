import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import NetworkDiagram from "./components/NetworkDiagram";
import ExpertUsage from "./components/ExpertUsage";
import TextInput from "./components/TextInput";
import InfoPanel from "./components/InfoPanel";
import {
  fetchModelInfo,
  runInference,
  type InferResponse,
  type ModelInfo,
} from "./lib/api";

export default function App() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [result, setResult] = useState<InferResponse | null>(null);
  const [selectedToken, setSelectedToken] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModelInfo()
      .then(setModelInfo)
      .catch(() => setError("Cannot reach backend — is it running on :8000?"));
  }, []);

  const handleInfer = useCallback(async (text: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await runInference(text, 64, 0.8);
      setResult(res);
      setSelectedToken(0);
    } catch {
      setError("Inference failed — check the backend logs.");
    } finally {
      setLoading(false);
    }
  }, []);

  const tokens = result?.input_tokens ?? [];
  const nExperts = modelInfo?.n_expert ?? 8;

  return (
    <div className="min-h-screen bg-[#0b0f1a] text-slate-200 flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-800/60 px-6 py-3 shrink-0">
        <div className="max-w-[1600px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 text-lg font-bold">
              ⚡
            </div>
            <div>
              <h1 className="text-lg font-bold text-white tracking-tight">Expert Lens</h1>
              <p className="text-[11px] text-slate-500">Mixture-of-Experts Visualizer</p>
            </div>
          </div>
          {modelInfo && (
            <div className="text-[11px] text-slate-500 font-mono">
              {(modelInfo.n_params / 1e6).toFixed(1)}M params · {modelInfo.n_layer} layers · {modelInfo.n_expert} experts · top-{modelInfo.top_k}
            </div>
          )}
        </div>
      </header>

      {/* Main content — everything visible at once */}
      <main className="flex-1 max-w-[1600px] w-full mx-auto px-6 py-4 flex flex-col gap-4 min-h-0">
        {/* Error banner */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2 text-sm text-red-400 shrink-0"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Top row: Input + Generated output */}
        <div className="flex gap-4 shrink-0">
          <div className="flex-1">
            <TextInput onSubmit={handleInfer} loading={loading} />
          </div>
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: "auto" }}
                className="flex-1 bg-slate-900/50 border border-slate-800/60 rounded-lg px-4 py-3 overflow-hidden"
              >
                <div className="text-xs text-slate-500 mb-1 font-medium">Generated</div>
                <p className="text-sm font-mono text-slate-300 leading-relaxed whitespace-pre-wrap max-h-20 overflow-y-auto">
                  <span className="text-indigo-400">{result.input_tokens.join("")}</span>
                  <span className="text-emerald-400">{result.generated_tokens.join("")}</span>
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Main visualization — single row, everything visible */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 flex-1 min-h-0">
          {/* Network Diagram — hero, takes most space */}
          <div className="lg:col-span-3">
            <Panel title="Network Architecture" subtitle="Active experts glow for the selected token" className="h-full flex flex-col">
              <div className="flex-1 min-h-0">
                <NetworkDiagram
                  modelInfo={modelInfo}
                  layers={result?.layers ?? null}
                  selectedToken={selectedToken}
                />
              </div>
              {/* Token selector */}
              {tokens.length > 0 && (
                <div className="mt-2 flex items-center gap-1 flex-wrap shrink-0">
                  <span className="text-[10px] text-slate-500 mr-1">Token:</span>
                  {tokens.map((tok, i) => {
                    const display = tok === " " ? "·" : tok === "\n" ? "↵" : tok;
                    return (
                      <button
                        key={`tok-${i}`}
                        onClick={() => setSelectedToken(i)}
                        className={`px-1.5 py-0.5 text-[11px] font-mono rounded cursor-pointer transition-colors ${
                          i === selectedToken
                            ? "bg-indigo-600 text-white"
                            : "bg-slate-800/60 text-slate-400 hover:bg-slate-700/60"
                        }`}
                      >
                        {display}
                      </button>
                    );
                  })}
                </div>
              )}
            </Panel>
          </div>

          {/* Right column: Expert Usage + About stacked */}
          <div className="flex flex-col gap-4">
            <Panel title="Expert Usage" subtitle="Activations per expert across all tokens">
              <ExpertUsage layers={result?.layers ?? null} nExperts={nExperts} />
            </Panel>
            <Panel title="What is MoE?" className="flex-1">
              <InfoPanel modelInfo={modelInfo} />
            </Panel>
          </div>
        </div>
      </main>
    </div>
  );
}

function Panel({
  title,
  subtitle,
  children,
  className = "",
}: {
  readonly title: string;
  readonly subtitle?: string;
  readonly children: React.ReactNode;
  readonly className?: string;
}) {
  return (
    <section className={`bg-slate-900/40 border border-slate-800/50 rounded-xl p-4 ${className}`}>
      <div className="mb-2">
        <h2 className="text-sm font-semibold text-slate-200">{title}</h2>
        {subtitle && <p className="text-[11px] text-slate-500 mt-0.5">{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}
