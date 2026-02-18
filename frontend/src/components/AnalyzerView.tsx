import { useCallback, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { fetchExpertProfile, type ExpertProfileResponse, type ModelInfo, type ModelType } from "../lib/api";
import AnalysisProgress from "./AnalysisProgress";

export interface AnalyzerState {
  modelType: ModelType;
  phase: Phase;
  profile: ExpertProfileResponse | null;
  selectedLayer: number;
}

const EXPERT_COLORS = [
  "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e",
  "#f97316", "#eab308", "#22c55e", "#06b6d4",
];

function getSteps(isBpe: boolean) {
  const unit = isBpe ? "token" : "character";
  return [
    { label: "Feed sample texts", desc: "150+ diverse sentences go through the model" },
    { label: "Record routing", desc: `Track which expert handles each ${unit}` },
    { label: "Find patterns", desc: "Discover what each expert specializes in" },
  ];
}

interface Props {
  readonly modelInfo: ModelInfo | null;
  readonly modelType: ModelType;
  readonly cached: AnalyzerState | null;
  readonly onCacheUpdate: (state: AnalyzerState) => void;
  readonly onError: (msg: string) => void;
}

type Phase = "idle" | "running" | "done";

export default function AnalyzerView({ modelInfo, modelType, cached, onCacheUpdate, onError }: Props) {
  const [phase, setPhase] = useState<Phase>(cached?.phase ?? "idle");
  const [activeStep, setActiveStep] = useState(cached?.phase === "done" ? 2 : -1);
  const [profile, setProfile] = useState<ExpertProfileResponse | null>(cached?.profile ?? null);
  const [selectedLayer, setSelectedLayer] = useState(cached?.selectedLayer ?? 0);

  // Persist state changes up to parent cache
  useEffect(() => {
    if (phase === "done" && profile) {
      onCacheUpdate({ modelType, phase, profile, selectedLayer });
    }
  }, [phase, profile, selectedLayer, modelType, onCacheUpdate]);

  const runAnalysis = useCallback(async () => {
    setPhase("running");
    setProfile(null);
    setActiveStep(0);

    const stepDelay = (ms: number) => new Promise((r) => setTimeout(r, ms));
    const request = fetchExpertProfile(modelType);

    // Step 0: Ingesting Corpus (show samples cycling through)
    await stepDelay(20000);
    
    // Step 1: Mapping Expert Activations (router animation)
    setActiveStep(1);
    await stepDelay(1500);
    
    // Step 2: LLM Profiling (waiting for response)
    setActiveStep(2);

    try {
      const res = await request;
      await stepDelay(400);
      setProfile(res);
      setSelectedLayer(0);
      setPhase("done");
    } catch {
      onError("Expert analysis failed — check the backend logs.");
      setPhase("idle");
      setActiveStep(-1);
    }
  }, [onError, modelType]);

  const nLayers = modelInfo?.n_layer ?? 3;
  const nExperts = modelInfo?.n_expert ?? 8;
  const demoChars = profile?.demo_layers[selectedLayer] ?? [];
  const layerExperts = profile?.layers[selectedLayer] ?? [];

  return (
    <div className="flex flex-col gap-5">
      {/* Top row: How it works + action */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 bg-slate-900/40 border border-slate-800/50 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-slate-200 mb-1">How Expert Profiling Works</h2>
          <p className="text-[11px] text-slate-500 mb-5">
            Each expert in a MoE model learns to handle certain types of characters.
            We discover this by running many sentences through the model and observing which expert the router assigns to each character.
          </p>

          {/* Pipeline steps */}
          <div className="flex items-start gap-2">
            {getSteps(modelType === "bpe").map((step, i) => {
              const isActive = phase === "running" && activeStep === i;
              const isDone = phase === "running" ? i < activeStep : phase === "done";

              const borderColor = isDone ? "#22c55e" : (isActive ? "#6366f1" : "#334155");
              const bgColor = isDone ? "#22c55e18" : (isActive ? "#6366f118" : "transparent");
              const textColor = isDone ? "#22c55e" : (isActive ? "#818cf8" : "#64748b");

              return (
                <div key={step.label} className="flex-1 flex items-start gap-2">
                  {i > 0 && (
                    <motion.div
                      className="w-8 h-0.5 mt-[18px] shrink-0"
                      animate={{ backgroundColor: isDone ? "#22c55e55" : "#1e293b" }}
                      transition={{ duration: 0.3 }}
                    />
                  )}
                  <div className="flex flex-col items-center text-center flex-1">
                    <div className="relative mb-2">
                      <motion.div
                        className="w-9 h-9 rounded-full border-2 flex items-center justify-center text-xs font-bold"
                        animate={{
                          borderColor,
                          backgroundColor: bgColor,
                          color: textColor,
                          scale: isActive ? 1.15 : 1,
                        }}
                        transition={{ duration: 0.3 }}
                      >
                        {isDone ? "✓" : i + 1}
                      </motion.div>
                      {isActive && (
                        <motion.div
                          className="absolute inset-0 rounded-full border-2 border-indigo-400"
                          initial={{ opacity: 0.6, scale: 1 }}
                          animate={{ opacity: 0, scale: 1.6 }}
                          transition={{ duration: 1, repeat: Infinity }}
                        />
                      )}
                    </div>
                    <div className="text-[11px] font-medium text-slate-300 mb-0.5">{step.label}</div>
                    <div className="text-[9px] text-slate-500 leading-tight">{step.desc}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Action card */}
        <div className="bg-slate-900/40 border border-slate-800/50 rounded-xl p-5 flex flex-col items-center justify-center text-center gap-4">
          <div>
            <div className="text-3xl mb-2">🔬</div>
            <h3 className="text-sm font-semibold text-slate-200">Expert Analyzer</h3>
            <p className="text-[11px] text-slate-500 mt-1">
              Profile {nExperts} experts across {nLayers} layers
            </p>
          </div>
          <button
            onClick={runAnalysis}
            disabled={phase === "running"}
            className="px-5 py-2.5 text-sm font-medium rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            {phase === "running" && (
              <span className="flex items-center gap-2">
                <span className="inline-block w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Analyzing…
              </span>
            )}
            {phase === "done" && "Re-analyze"}
            {phase === "idle" && "Run Analysis"}
          </button>
          {phase === "done" && profile && (
            <div className="text-[10px] text-emerald-500">
              ✓ {profile.sample_count} samples profiled
            </div>
          )}
        </div>
      </div>

      {/* Warnings */}
      <AnimatePresence>
        {profile?.warnings && profile.warnings.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-amber-500/10 border border-amber-500/30 rounded-lg px-4 py-3"
          >
            <div className="flex items-start gap-3">
              <span className="text-xl">⚠️</span>
              <div>
                <h3 className="text-sm font-semibold text-amber-400">LLM Labeling Issues</h3>
                <ul className="mt-1 space-y-1">
                  {profile.warnings.map((w, i) => (
                    <li key={i} className="text-[11px] text-amber-300/80 font-mono">
                      {w}
                    </li>
                  ))}
                </ul>
                <p className="text-[11px] text-amber-400/60 mt-2">
                  Showing programmatic labels instead. Check your LITELLM_URL and connectivity.
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Progress Visualization */}
      <AnimatePresence mode="wait">
        {phase === "running" && (
          <motion.div
            key="progress"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <AnalysisProgress step={activeStep} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results */}
      <AnimatePresence>
        {phase === "done" && profile && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.4 }}
            className="flex flex-col gap-5"
          >
            {/* Layer tabs */}
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-slate-400 mr-2">Layer:</span>
              {profile.layers.map((_, li) => (
                <button
                  key={`lt-${li}`}
                  onClick={() => setSelectedLayer(li)}
                  className={`px-3 py-1.5 text-[11px] font-medium rounded-lg transition-colors cursor-pointer ${
                    li === selectedLayer
                      ? "bg-indigo-600 text-white"
                      : "bg-slate-800/60 text-slate-400 hover:bg-slate-700/60"
                  }`}
                >
                  Layer {li}
                </button>
              ))}
            </div>

            {/* Color-coded sentence */}
            <div className="bg-slate-900/40 border border-slate-800/50 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-slate-200 mb-1">
                Who handles what?
              </h2>
              <p className="text-[11px] text-slate-500 mb-4">
                {modelType === "bpe"
                  ? "Each token (subword) is colored by the expert that handles it. BPE tokens carry more meaning than single characters."
                  : "Each character is colored by the expert that handles it. Notice how different experts claim different parts of the sentence."}
              </p>
              <div className="bg-slate-950/50 rounded-lg p-4 mb-4">
                <span className={`font-mono leading-relaxed ${modelType === "bpe" ? "text-base" : "text-2xl tracking-wide"}`}>
                  {demoChars.map((dc, i) => {
                    const color = dc.expert >= 0
                      ? EXPERT_COLORS[dc.expert % EXPERT_COLORS.length]
                      : "#475569";
                    const isBpeToken = modelType === "bpe";
                    let display = dc.char;
                    if (dc.char === " ") {
                      display = isBpeToken ? "·" : "\u00A0";
                    }
                    return (
                      <motion.span
                        key={`dc-${selectedLayer}-${i}`}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3, delay: i * 0.015 }}
                        className={`relative inline-block ${isBpeToken ? "px-1 py-0.5 mx-0.5 mb-1 rounded text-[12px]" : ""}`}
                        style={{
                          color: isBpeToken ? "#e2e8f0" : color,
                          backgroundColor: isBpeToken ? `${color}30` : "transparent",
                          borderBottom: isBpeToken ? `2px solid ${color}` : "none",
                        }}
                        title={`Expert ${dc.expert}`}
                      >
                        {!isBpeToken && (
                          <span
                            className="absolute -bottom-1 left-0 right-0 h-0.5 rounded-full"
                            style={{ backgroundColor: color, opacity: 0.5 }}
                          />
                        )}
                        {display}
                      </motion.span>
                    );
                  })}
                </span>
              </div>
              {/* Legend */}
              <div className="flex flex-wrap gap-3">
                {layerExperts.map((expert) => {
                  const color = EXPERT_COLORS[expert.expert_id % EXPERT_COLORS.length];
                  return (
                    <div key={`leg-${expert.expert_id}`} className="flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: color }} />
                      <span className="text-[10px] text-slate-400">
                        E{expert.expert_id}
                      </span>
                      <span className="text-[10px] font-medium" style={{ color }}>
                        {expert.role}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Expert cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {layerExperts.map((expert) => {
                const color = EXPERT_COLORS[expert.expert_id % EXPERT_COLORS.length];
                return (
                  <motion.div
                    key={`e-${selectedLayer}-${expert.expert_id}`}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: expert.expert_id * 0.04 }}
                    className="rounded-xl border p-4 flex flex-col gap-3"
                    style={{
                      borderColor: `${color}30`,
                      background: `linear-gradient(135deg, ${color}06 0%, transparent 50%)`,
                    }}
                  >
                    {/* Role name + description */}
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <div
                          className="w-3 h-3 rounded-full shrink-0"
                          style={{ backgroundColor: color }}
                        />
                        <span className="text-[11px] font-medium text-slate-400">
                          Expert {expert.expert_id}
                        </span>
                      </div>
                      <div
                        className="text-base font-bold leading-snug"
                        style={{ color }}
                      >
                        {expert.role}
                      </div>
                      {expert.domain && (
                        <span
                          className="inline-block text-[9px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full mt-1"
                          style={{
                            backgroundColor: `${color}15`,
                            color,
                            border: `1px solid ${color}25`,
                          }}
                        >
                          {expert.domain}
                        </span>
                      )}
                      {expert.description && (
                        <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">
                          {expert.description}
                        </p>
                      )}
                    </div>

                    {/* Top characters */}
                    <div className="flex gap-1.5">
                      {expert.top_chars.slice(0, 5).map((c) => {
                        const display = c.char === " " ? "␣" : c.char;
                        return (
                          <div
                            key={`ch-${expert.expert_id}-${c.char}`}
                            className="w-8 h-8 rounded-md flex items-center justify-center text-sm font-mono font-bold"
                            style={{
                              backgroundColor: `${color}18`,
                              color,
                              border: `1px solid ${color}30`,
                            }}
                            title={`"${c.char}" — ${c.pct}% of this expert's work`}
                          >
                            {display}
                          </div>
                        );
                      })}
                    </div>

                    {/* Highlighted example words */}
                    {expert.example_words.length > 0 && (
                      <div>
                        <div className="text-[9px] text-slate-600 mb-1.5">
                          Handles the highlighted letters:
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {expert.example_words.slice(0, 4).map((ew) => (
                            <span
                              key={`ew-${expert.expert_id}-${ew.word}`}
                              className="text-sm font-mono tracking-wide"
                            >
                              {ew.word.split("").map((ch, ci) => (
                                <span
                                  key={`ewc-${expert.expert_id}-${ew.word}-${ci}`}
                                  className="inline-block"
                                  style={{
                                    color: ew.highlights[ci] ? color : "#475569",
                                    fontWeight: ew.highlights[ci] ? 700 : 400,
                                    textDecoration: ew.highlights[ci] ? "underline" : "none",
                                    textDecorationColor: ew.highlights[ci] ? `${color}60` : "transparent",
                                    textUnderlineOffset: "3px",
                                  }}
                                >
                                  {ch}
                                </span>
                              ))}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty state */}
      {phase === "idle" && (
        <div className="flex-1 flex items-center justify-center py-16">
          <div className="text-center text-slate-600">
            <div className="text-4xl mb-3 opacity-40">🧠</div>
            <p className="text-sm">Click "Run Analysis" to discover what each expert has learned</p>
          </div>
        </div>
      )}
    </div>
  );
}
