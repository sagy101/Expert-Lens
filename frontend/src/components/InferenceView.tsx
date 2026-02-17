import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import NetworkDiagram from "./NetworkDiagram";
import ExpertUsage from "./ExpertUsage";
import TextInput from "./TextInput";
import InfoPanel from "./InfoPanel";
import { runInference, type InferResponse, type ModelInfo, type ModelType } from "../lib/api";

export interface InferenceState {
  modelType: ModelType;
  result: InferResponse | null;
  selectedToken: number;
}

interface Props {
  readonly modelInfo: ModelInfo | null;
  readonly modelType: ModelType;
  readonly cached: InferenceState | null;
  readonly onCacheUpdate: (state: InferenceState) => void;
  readonly onError: (msg: string) => void;
}

export default function InferenceView({ modelInfo, modelType, cached, onCacheUpdate, onError }: Props) {
  const [result, setResult] = useState<InferResponse | null>(cached?.result ?? null);
  const [selectedToken, setSelectedToken] = useState(cached?.selectedToken ?? 0);
  const [loading, setLoading] = useState(false);
  const [justCompleted, setJustCompleted] = useState(false);
  const [resultKey, setResultKey] = useState(0);

  useEffect(() => {
    if (result) {
      onCacheUpdate({ modelType, result, selectedToken });
    }
  }, [result, selectedToken, modelType, onCacheUpdate]);

  const handleInfer = useCallback(async (text: string) => {
    setLoading(true);
    setJustCompleted(false);
    try {
      const res = await runInference(text, 64, 0.8, modelType);
      setResult(res);
      setSelectedToken(0);
      setResultKey((k) => k + 1);
      setJustCompleted(true);
      setTimeout(() => setJustCompleted(false), 1200);
    } catch {
      onError("Inference failed — check the backend logs.");
    } finally {
      setLoading(false);
    }
  }, [onError, modelType]);

  const tokens = result?.input_tokens ?? [];
  const nExperts = modelInfo?.n_expert ?? 8;

  return (
    <>
      {/* Top row: Input + Generated output */}
      <div className="flex gap-4 shrink-0">
        <div className="flex-1">
          <TextInput onSubmit={handleInfer} loading={loading} />
        </div>
        <AnimatePresence>
          {result && (
            <motion.div
              key={`gen-${resultKey}`}
              initial={{ opacity: 0, scale: 0.95, x: 20 }}
              animate={{ opacity: 1, scale: 1, x: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="flex-1 bg-slate-900/50 border border-slate-800/60 rounded-lg px-4 py-3 overflow-hidden relative"
            >
              {/* Success flash */}
              {justCompleted && (
                <motion.div
                  className="absolute inset-0 rounded-lg pointer-events-none"
                  initial={{ opacity: 0.3 }}
                  animate={{ opacity: 0 }}
                  transition={{ duration: 1.2 }}
                  style={{ background: "radial-gradient(ellipse at center, rgba(99,102,241,0.15), transparent 70%)" }}
                />
              )}
              <div className="text-xs text-slate-500 mb-1 font-medium flex items-center gap-2">
                Generated
                {justCompleted && (
                  <motion.span
                    initial={{ opacity: 1, scale: 1 }}
                    animate={{ opacity: 0, scale: 0.8 }}
                    transition={{ duration: 1, delay: 0.3 }}
                    className="text-emerald-400 text-[10px]"
                  >
                    ✓
                  </motion.span>
                )}
              </div>
              <p className="text-sm font-mono text-slate-300 leading-relaxed whitespace-pre-wrap max-h-20 overflow-y-auto">
                <span className="text-indigo-400">{result.input_tokens.join("")}</span>
                <motion.span
                  key={`tok-${resultKey}`}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.6, delay: 0.15 }}
                  className="text-emerald-400"
                >
                  {result.generated_tokens.join("")}
                </motion.span>
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Parameter efficiency bar */}
      <AnimatePresence>
        {result && modelInfo && (
          <motion.div
            key={`params-${resultKey}`}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="bg-slate-900/40 border border-slate-800/50 rounded-lg px-4 py-3 shrink-0"
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] font-medium text-slate-400">
                Parameters used per token
              </span>
              <span className="text-[11px] text-slate-500 font-mono">
                {(modelInfo.n_active_params / 1e3).toFixed(1)}K of {(modelInfo.n_params / 1e6).toFixed(1)}M
                ({((modelInfo.n_active_params / modelInfo.n_params) * 100).toFixed(0)}%)
              </span>
            </div>
            <div className="relative h-5 rounded-full bg-slate-800/60 overflow-hidden">
              {/* Full bar background with subtle tick marks */}
              <div className="absolute inset-0 flex items-center px-1">
                {Array.from({ length: 10 }, (_, i) => (
                  <div
                    key={`tick-${i}`}
                    className="flex-1 border-r border-slate-700/30 h-full"
                  />
                ))}
              </div>
              {/* Active params bar */}
              <motion.div
                className="absolute inset-y-0 left-0 rounded-full"
                style={{
                  background: "linear-gradient(90deg, #22c55e, #4ade80)",
                }}
                initial={{ width: 0 }}
                animate={{
                  width: `${(modelInfo.n_active_params / modelInfo.n_params) * 100}%`,
                }}
                transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
              />
              {/* Label inside bar */}
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-[9px] font-bold text-white drop-shadow-sm">
                  Only {modelInfo.top_k} of {modelInfo.n_expert} experts active — {((1 - modelInfo.n_active_params / modelInfo.n_params) * 100).toFixed(0)}% compute saved
                </span>
              </div>
            </div>
            <p className="text-[10px] text-slate-600 mt-1">
              MoE activates only {modelInfo.top_k} out of {modelInfo.n_expert} experts per token, using a fraction of the total network while keeping full model capacity.
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 flex-1 min-h-0">
        <motion.div
          key={`net-${resultKey}`}
          className="lg:col-span-3"
          initial={resultKey > 0 ? { opacity: 0.6 } : false}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <Panel title="Network Architecture" subtitle="Active experts glow for the selected token" className="h-full flex flex-col">
            <div className="flex-1 min-h-0">
              <NetworkDiagram
                modelInfo={modelInfo}
                layers={result?.layers ?? null}
                selectedToken={selectedToken}
              />
            </div>
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
        </motion.div>

        <motion.div
          key={`side-${resultKey}`}
          className="flex flex-col gap-4"
          initial={resultKey > 0 ? { opacity: 0, y: 10 } : false}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.15 }}
        >
          <Panel title="Expert Usage" subtitle="Activations per expert across all tokens">
            <ExpertUsage layers={result?.layers ?? null} nExperts={nExperts} />
          </Panel>
          <Panel title="What is MoE?" className="flex-1">
            <InfoPanel modelInfo={modelInfo} />
          </Panel>
        </motion.div>
      </div>
    </>
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
