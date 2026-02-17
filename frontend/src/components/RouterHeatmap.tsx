import { motion } from "framer-motion";
import type { LayerInfo } from "../lib/api";

const HEAT_COLORS = [
  "rgba(30, 27, 75, 0.9)",   // 0%   — near black
  "rgba(67, 56, 202, 0.9)",  // 20%
  "rgba(139, 92, 246, 0.9)", // 40%
  "rgba(236, 72, 153, 0.9)", // 60%
  "rgba(249, 115, 22, 0.9)", // 80%
  "rgba(234, 179, 8, 0.9)",  // 100%
];

function probToColor(p: number): string {
  const idx = Math.min(Math.floor(p * (HEAT_COLORS.length - 1)), HEAT_COLORS.length - 2);
  const t = p * (HEAT_COLORS.length - 1) - idx;
  return HEAT_COLORS[Math.round(idx + t)];
}

interface Props {
  layers: LayerInfo[] | null;
  tokens: string[];
  selectedToken: number;
  onSelectToken: (idx: number) => void;
  nExperts: number;
}

export default function RouterHeatmap({
  layers,
  tokens,
  selectedToken,
  onSelectToken,
  nExperts,
}: Props) {
  if (!layers || tokens.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-500 text-sm">
        Run inference to see router probabilities
      </div>
    );
  }

  const cellW = Math.max(18, Math.min(32, 600 / tokens.length));
  const cellH = 22;
  const labelW = 54;

  return (
    <div className="overflow-x-auto">
      {layers.map((layer, li) => (
        <div key={li} className="mb-4">
          <div className="text-xs text-slate-400 mb-1 font-medium">Layer {li}</div>
          <div className="flex">
            {/* Expert labels */}
            <div className="flex flex-col" style={{ minWidth: labelW }}>
              <div style={{ height: cellH }} />
              {Array.from({ length: nExperts }, (_, ei) => (
                <div
                  key={ei}
                  className="text-[10px] text-slate-500 flex items-center justify-end pr-1"
                  style={{ height: cellH }}
                >
                  E{ei}
                </div>
              ))}
            </div>

            {/* Grid */}
            <div>
              {/* Token headers */}
              <div className="flex">
                {tokens.map((tok, ti) => (
                  <button
                    key={ti}
                    className={`text-[10px] flex items-center justify-center cursor-pointer transition-colors ${
                      ti === selectedToken
                        ? "text-indigo-300 font-bold"
                        : "text-slate-500"
                    }`}
                    style={{ width: cellW, height: cellH }}
                    onClick={() => onSelectToken(ti)}
                  >
                    {tok === " " ? "·" : tok === "\n" ? "↵" : tok}
                  </button>
                ))}
              </div>

              {/* Heatmap cells */}
              {Array.from({ length: nExperts }, (_, ei) => (
                <div key={ei} className="flex">
                  {tokens.map((_, ti) => {
                    const prob = layer.router_probs[ti]?.[ei] ?? 0;
                    const isSelected = layer.topk_indices[ti]?.includes(ei);
                    return (
                      <motion.div
                        key={ti}
                        className={`border cursor-pointer transition-all ${
                          ti === selectedToken
                            ? "border-indigo-400/60"
                            : "border-slate-800/50"
                        }`}
                        style={{
                          width: cellW,
                          height: cellH,
                          backgroundColor: probToColor(prob),
                        }}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: ti * 0.005 + ei * 0.01 }}
                        onClick={() => onSelectToken(ti)}
                        title={`Token "${tokens[ti]}" → Expert ${ei}: ${(prob * 100).toFixed(1)}%${isSelected ? " ★" : ""}`}
                      >
                        {isSelected && (
                          <div className="w-full h-full flex items-center justify-center">
                            <div
                              className="rounded-full"
                              style={{
                                width: 5,
                                height: 5,
                                backgroundColor: "#fff",
                                opacity: 0.8,
                              }}
                            />
                          </div>
                        )}
                      </motion.div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
      ))}

      {/* Legend */}
      <div className="flex items-center gap-2 mt-2 text-[10px] text-slate-500">
        <span>0%</span>
        <div className="flex h-3 rounded overflow-hidden">
          {HEAT_COLORS.map((c, i) => (
            <div key={i} style={{ backgroundColor: c, width: 24, height: 12 }} />
          ))}
        </div>
        <span>100%</span>
        <span className="ml-2 text-slate-400">● = selected expert</span>
      </div>
    </div>
  );
}
