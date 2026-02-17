import { motion } from "framer-motion";
import type { LayerInfo } from "../lib/api";

const EXPERT_COLORS = [
  "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e",
  "#f97316", "#eab308", "#22c55e", "#06b6d4",
];

interface Props {
  readonly layers: LayerInfo[] | null;
  readonly nExperts: number;
}

export default function ExpertUsage({ layers, nExperts }: Props) {
  if (!layers) {
    return (
      <div className="flex items-center justify-center h-32 text-slate-500 text-sm">
        Run inference to see expert usage
      </div>
    );
  }

  const counts: number[][] = layers.map((layer) => {
    const c = new Array(nExperts).fill(0);
    for (const row of layer.topk_indices) {
      for (const idx of row) {
        c[idx]++;
      }
    }
    return c;
  });

  const maxCount = Math.max(...counts.flat(), 1);

  return (
    <div className="space-y-4">
      {counts.map((layerCounts, li) => {
        const total = layerCounts.reduce((a, b) => a + b, 0);
        return (
          <div key={`layer-${li}`}>
            <div className="text-xs text-slate-400 mb-1.5 font-medium">Layer {li}</div>
            <div className="flex items-end gap-1" style={{ height: 64 }}>
              {layerCounts.map((count, ei) => {
                const pct = count / maxCount;
                const usagePct = total > 0 ? ((count / total) * 100).toFixed(1) : "0";
                return (
                  <div
                    key={`e-${li}-${ei}`}
                    className="flex flex-col items-center flex-1"
                    title={`Expert ${ei}: ${count} activations (${usagePct}%)`}
                  >
                    <motion.div
                      className="w-full rounded-t"
                      style={{
                        backgroundColor: EXPERT_COLORS[ei % EXPERT_COLORS.length],
                        minWidth: 12,
                      }}
                      initial={{ height: 0 }}
                      animate={{ height: Math.max(2, pct * 52) }}
                      transition={{ duration: 0.5, delay: ei * 0.03 }}
                    />
                    <span className="text-[9px] text-slate-500 mt-0.5">E{ei}</span>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
