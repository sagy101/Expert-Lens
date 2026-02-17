import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { ExpertProfileResponse } from "../lib/api";

const EXPERT_COLORS = [
  "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e",
  "#f97316", "#eab308", "#22c55e", "#06b6d4",
];

interface Props {
  readonly profile: ExpertProfileResponse | null;
  readonly loading: boolean;
  readonly onAnalyze: () => void;
}

export default function ExpertIdentity({ profile, loading, onAnalyze }: Props) {
  const [selectedLayer, setSelectedLayer] = useState(0);

  if (!profile) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-6">
        <p className="text-slate-500 text-sm text-center">
          Discover what each expert specializes in by profiling routing decisions across sample texts.
        </p>
        <button
          onClick={onAnalyze}
          disabled={loading}
          className="px-4 py-2 text-sm font-medium rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <span className="inline-block w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Profiling…
            </span>
          ) : (
            "Profile Experts"
          )}
        </button>
      </div>
    );
  }

  const layerData = profile.layers[selectedLayer] ?? [];

  return (
    <div className="space-y-3">
      {/* Layer tabs */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-slate-500 mr-1">Layer:</span>
        {profile.layers.map((_, li) => (
          <button
            key={`lt-${li}`}
            onClick={() => setSelectedLayer(li)}
            className={`px-2.5 py-1 text-[11px] font-medium rounded transition-colors cursor-pointer ${
              li === selectedLayer
                ? "bg-indigo-600 text-white"
                : "bg-slate-800/60 text-slate-400 hover:bg-slate-700/60"
            }`}
          >
            Layer {li}
          </button>
        ))}
        <span className="ml-auto text-[10px] text-slate-600">
          {profile.sample_count} samples
        </span>
      </div>

      {/* Expert cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <AnimatePresence mode="wait">
          {layerData.map((expert) => {
            const color = EXPERT_COLORS[expert.expert_id % EXPERT_COLORS.length];
            return (
              <motion.div
                key={`e-${selectedLayer}-${expert.expert_id}`}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.25, delay: expert.expert_id * 0.03 }}
                className="rounded-lg border p-2.5 flex flex-col gap-2 relative overflow-hidden"
                style={{
                  borderColor: `${color}30`,
                  background: `linear-gradient(135deg, ${color}08 0%, transparent 60%)`,
                }}
              >
                {/* Expert ID + label */}
                <div>
                  <div className="flex items-center gap-1.5 mb-1">
                    <div
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-[11px] font-bold text-slate-300">
                      E{expert.expert_id}
                    </span>
                  </div>
                  <div
                    className="text-[11px] font-semibold leading-tight"
                    style={{ color }}
                  >
                    {expert.label}
                  </div>
                </div>

                {/* Tag breakdown */}
                <div className="space-y-0.5">
                  {expert.top_tags.slice(0, 4).map((tag) => (
                    <div key={`tag-${expert.expert_id}-${tag.tag}`} className="flex items-center gap-1.5">
                      <div className="flex-1 h-1 rounded-full bg-slate-800/60 overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.min(tag.pct, 100)}%`,
                            backgroundColor: color,
                            opacity: 0.7,
                          }}
                        />
                      </div>
                      <span className="text-[9px] text-slate-500 w-[90px] text-right truncate" title={tag.tag}>
                        {tag.tag} {tag.pct}%
                      </span>
                    </div>
                  ))}
                </div>

                {/* Top words */}
                {expert.top_words.length > 0 && (
                  <div className="flex flex-wrap gap-0.5">
                    {expert.top_words.slice(0, 6).map((w) => (
                      <span
                        key={`w-${expert.expert_id}-${w.word}`}
                        className="text-[9px] font-mono px-1.5 py-0.5 rounded"
                        style={{
                          backgroundColor: `${color}18`,
                          color,
                        }}
                        title={`"${w.word}" — ${w.count}×`}
                      >
                        {w.word}
                      </span>
                    ))}
                  </div>
                )}

                {/* Activation count */}
                <div className="text-[9px] text-slate-600 mt-auto">
                  {expert.total_activations} activations
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* Re-analyze */}
      <div className="flex justify-end">
        <button
          onClick={onAnalyze}
          disabled={loading}
          className="text-[11px] text-slate-500 hover:text-slate-300 transition-colors disabled:opacity-50 cursor-pointer"
        >
          {loading ? "Profiling…" : "Re-analyze"}
        </button>
      </div>
    </div>
  );
}
