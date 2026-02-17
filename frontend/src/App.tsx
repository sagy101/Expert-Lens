import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import InferenceView, { type InferenceState } from "./components/InferenceView";
import AnalyzerView, { type AnalyzerState } from "./components/AnalyzerView";
import { fetchModelInfo, type ModelInfo, type ModelType } from "./lib/api";

type Tab = "inference" | "analyzer";

export default function App() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("inference");
  const [modelType, setModelType] = useState<ModelType>("char");
  const [inferenceCache, setInferenceCache] = useState<Partial<Record<ModelType, InferenceState>>>({});
  const [analyzerCache, setAnalyzerCache] = useState<Partial<Record<ModelType, AnalyzerState>>>({});

  const savedInference = inferenceCache[modelType] ?? null;
  const handleInferenceChange = useCallback((state: InferenceState) => {
    setInferenceCache((prev) => ({ ...prev, [state.modelType]: state }));
  }, []);

  const savedAnalyzer = analyzerCache[modelType] ?? null;
  const handleAnalyzerChange = useCallback((state: AnalyzerState) => {
    setAnalyzerCache((prev) => ({ ...prev, [state.modelType]: state }));
  }, []);

  useEffect(() => {
    fetchModelInfo(modelType)
      .then(setModelInfo)
      .catch(() => setError("Cannot reach backend — is it running on :8000?"));
  }, [modelType]);

  const handleError = useCallback((msg: string) => setError(msg), []);

  const TABS: { id: Tab; label: string; icon: string }[] = [
    { id: "inference", label: "Inference", icon: "⚡" },
    { id: "analyzer", label: "Expert Analyzer", icon: "🔬" },
  ];

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

          {/* Tabs */}
          <div className="flex items-center gap-1 bg-slate-900/60 rounded-lg p-1 border border-slate-800/40">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`relative px-4 py-1.5 text-[12px] font-medium rounded-md transition-colors cursor-pointer ${
                  tab === t.id
                    ? "text-white"
                    : "text-slate-400 hover:text-slate-300"
                }`}
              >
                {tab === t.id && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-indigo-600/90 rounded-md"
                    transition={{ type: "spring", duration: 0.4, bounce: 0.15 }}
                  />
                )}
                <span className="relative flex items-center gap-1.5">
                  <span>{t.icon}</span>
                  {t.label}
                </span>
              </button>
            ))}
          </div>

          <div className="flex items-center gap-3">
            {/* Model type toggle */}
            {modelInfo && (modelInfo.available_models?.length ?? 0) > 1 && (
              <div className="flex items-center gap-1 bg-slate-900/60 rounded-lg p-0.5 border border-slate-800/40">
                {modelInfo.available_models.map((mt) => (
                  <button
                    key={mt}
                    onClick={() => setModelType(mt)}
                    className={`relative px-3 py-1 text-[10px] font-semibold rounded-md transition-colors cursor-pointer uppercase tracking-wider ${
                      modelType === mt
                        ? "text-white bg-emerald-600/90"
                        : "text-slate-500 hover:text-slate-300"
                    }`}
                  >
                    {mt === "char" ? "Character" : "BPE Tokens"}
                  </button>
                ))}
              </div>
            )}
            {modelInfo && (
              <div className="text-[11px] text-slate-500 font-mono">
                {(modelInfo.n_params / 1e6).toFixed(1)}M params · {modelInfo.n_layer} layers · {modelInfo.n_expert} experts · top-{modelInfo.top_k}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main content */}
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

        {/* Tab content */}
        <AnimatePresence mode="wait">
          {tab === "inference" ? (
            <motion.div
              key="inference"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.25 }}
              className="flex flex-col gap-4 flex-1 min-h-0"
            >
              <InferenceView key={modelType} modelInfo={modelInfo} modelType={modelType} cached={savedInference} onCacheUpdate={handleInferenceChange} onError={handleError} />
            </motion.div>
          ) : (
            <motion.div
              key="analyzer"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.25 }}
              className="flex flex-col gap-4 flex-1 min-h-0"
            >
              <AnalyzerView key={modelType} modelInfo={modelInfo} modelType={modelType} cached={savedAnalyzer} onCacheUpdate={handleAnalyzerChange} onError={handleError} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
