import { motion } from "framer-motion";
import type { LayerInfo, ModelInfo } from "../lib/api";

const EXPERT_COLORS = [
  "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e",
  "#f97316", "#eab308", "#22c55e", "#06b6d4",
];

interface Props {
  readonly modelInfo: ModelInfo | null;
  readonly layers: LayerInfo[] | null;
  readonly selectedToken: number;
}

/* ── layout constants ─────────────────────────────── */
const BLOCK_W = 210;
const BLOCK_GAP = 24;
const PAD_X = 56;
const PAD_Y = 18;

const ATTN_H = 36;
const ROUTER_H = 28;
const EXPERT_W = 40;
const EXPERT_H = 44;
const EXPERT_GAP = 6;
const SUM_R = 12;

export default function NetworkDiagram({ modelInfo, layers, selectedToken }: Props) {
  const nLayers = modelInfo?.n_layer ?? 3;
  const nExperts = modelInfo?.n_expert ?? 8;
  const topK = modelInfo?.top_k ?? 2;

  const expertCols = 4;
  const expertRows = Math.ceil(nExperts / expertCols);
  const expertsW = expertCols * EXPERT_W + (expertCols - 1) * EXPERT_GAP;
  const expertsH = expertRows * EXPERT_H + (expertRows - 1) * EXPERT_GAP;

  const blockInnerH = ATTN_H + 14 + ROUTER_H + 14 + expertsH + 14 + SUM_R * 2;
  const svgW = nLayers * BLOCK_W + (nLayers - 1) * BLOCK_GAP + PAD_X * 2 + 60;
  const svgH = blockInnerH + PAD_Y * 2 + 50;

  const getActiveExperts = (li: number): Set<number> => {
    if (!layers?.[li] || selectedToken < 0) return new Set();
    const row = layers[li].topk_indices[selectedToken];
    return row ? new Set(row) : new Set();
  };

  const getRouterProb = (li: number, ei: number): number => {
    if (!layers?.[li] || selectedToken < 0) return 0;
    const row = layers[li].router_probs[selectedToken];
    return row ? row[ei] : 0;
  };

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full h-full" style={{ minHeight: 340 }}>
      <defs>
        <filter id="glow-active">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <marker id="arrow" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#475569" />
        </marker>
      </defs>

      {/* ── Input label ─────────────────────────── */}
      <text x={PAD_X + 12} y={svgH / 2 + 4} textAnchor="end" fill="#818cf8" fontSize={11} fontWeight={600}>
        tokens →
      </text>

      {/* ── Output label ────────────────────────── */}
      <text x={svgW - PAD_X - 8} y={svgH / 2 + 4} textAnchor="start" fill="#818cf8" fontSize={11} fontWeight={600}>
        → logits
      </text>

      {/* ── Connector arrows between blocks ─────── */}
      {Array.from({ length: nLayers - 1 }, (_, i) => {
        const x1 = PAD_X + 20 + (i + 1) * BLOCK_W + i * BLOCK_GAP;
        const x2 = x1 + BLOCK_GAP;
        const cy = svgH / 2;
        return (
          <line key={`conn-${i}`} x1={x1} y1={cy} x2={x2} y2={cy}
            stroke="#475569" strokeWidth={1.5} markerEnd="url(#arrow)" />
        );
      })}

      {/* ── Transformer blocks ──────────────────── */}
      {Array.from({ length: nLayers }, (_, li) => {
        const bx = PAD_X + 20 + li * (BLOCK_W + BLOCK_GAP);
        const by = PAD_Y + 20;
        const active = getActiveExperts(li);

        let cy = by + 12;
        const attnY = cy;
        cy += ATTN_H + 14;
        const routerY = cy;
        cy += ROUTER_H + 14;
        const expertsY = cy;
        cy += expertsH + 14;
        const sumY = cy + SUM_R;

        const centerX = bx + BLOCK_W / 2;
        const expertsStartX = centerX - expertsW / 2;

        return (
          <g key={`block-${li}`}>
            {/* Block background */}
            <rect x={bx} y={by - 8} width={BLOCK_W} height={blockInnerH + 28}
              rx={10} fill="#0f172a" stroke="#1e293b" strokeWidth={1.5} />

            {/* Block title */}
            <text x={centerX} y={by + 2} textAnchor="middle" fill="#94a3b8" fontSize={10} fontWeight={600}>
              Transformer Block {li}
            </text>

            {/* ── Multi-Head Attention ──────────── */}
            <rect x={bx + 16} y={attnY} width={BLOCK_W - 32} height={ATTN_H}
              rx={6} fill="#1e1b4b" stroke="#4338ca" strokeWidth={1.2} />
            <text x={centerX} y={attnY + 15} textAnchor="middle" fill="#a5b4fc" fontSize={9} fontWeight={600}>
              Multi-Head Attention
            </text>
            <text x={centerX} y={attnY + 27} textAnchor="middle" fill="#6366f1" fontSize={7}>
              {modelInfo?.n_head ?? 6} heads · LayerNorm
            </text>

            {/* Residual skip arrow (attention) */}
            <line x1={bx + 10} y1={attnY - 4} x2={bx + 10} y2={attnY + ATTN_H + 8}
              stroke="#334155" strokeWidth={1} strokeDasharray="3 2" />
            <text x={bx + 7} y={attnY + ATTN_H + 14} textAnchor="middle" fill="#475569" fontSize={8}>+</text>

            {/* Arrow: attention → router */}
            <line x1={centerX} y1={attnY + ATTN_H + 2} x2={centerX} y2={routerY - 2}
              stroke="#475569" strokeWidth={1} markerEnd="url(#arrow)" />

            {/* ── Router ───────────────────────── */}
            <motion.rect x={bx + 30} y={routerY} width={BLOCK_W - 60} height={ROUTER_H}
              rx={14} fill={layers ? "#1e1b4b" : "#0f172a"} stroke="#7c3aed" strokeWidth={1.5}
              initial={{ opacity: 0.5 }} animate={{ opacity: 1 }} />
            <text x={centerX} y={routerY + 12} textAnchor="middle" fill="#c4b5fd" fontSize={9} fontWeight={600}>
              Router
            </text>
            <text x={centerX} y={routerY + 23} textAnchor="middle" fill="#7c3aed" fontSize={7}>
              softmax → top-{topK}
            </text>

            {/* ── Lines from router to experts ──── */}
            {Array.from({ length: nExperts }, (_, ei) => {
              const col = ei % expertCols;
              const row = Math.floor(ei / expertCols);
              const ex = expertsStartX + col * (EXPERT_W + EXPERT_GAP) + EXPERT_W / 2;
              const ey = expertsY + row * (EXPERT_H + EXPERT_GAP);
              const isActive = active.has(ei);
              const prob = getRouterProb(li, ei);
              const color = EXPERT_COLORS[ei % EXPERT_COLORS.length];

              return (
                <g key={`route-${li}-${ei}`}>
                  <motion.line
                    x1={centerX} y1={routerY + ROUTER_H}
                    x2={ex} y2={ey}
                    stroke={isActive ? color : "#1e293b"}
                    strokeWidth={isActive ? 2 : 0.5}
                    strokeOpacity={isActive ? 0.8 : 0.2}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3, delay: li * 0.1 + ei * 0.02 }}
                  />
                  {isActive && prob > 0 && (
                    <motion.text
                      x={(centerX + ex) / 2 + (col < expertCols / 2 ? -8 : 8)}
                      y={(routerY + ROUTER_H + ey) / 2}
                      textAnchor="middle" fill={color} fontSize={7} fontWeight={700}
                      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                      transition={{ delay: 0.3 }}
                    >
                      {(prob * 100).toFixed(0)}%
                    </motion.text>
                  )}
                </g>
              );
            })}

            {/* ── Expert MLPs ──────────────────── */}
            {Array.from({ length: nExperts }, (_, ei) => {
              const col = ei % expertCols;
              const row = Math.floor(ei / expertCols);
              const ex = expertsStartX + col * (EXPERT_W + EXPERT_GAP);
              const ey = expertsY + row * (EXPERT_H + EXPERT_GAP);
              const isActive = active.has(ei);
              const color = EXPERT_COLORS[ei % EXPERT_COLORS.length];

              return (
                <g key={`expert-${li}-${ei}`}>
                  {/* Expert outer box */}
                  <motion.rect
                    x={ex} y={ey} width={EXPERT_W} height={EXPERT_H} rx={4}
                    fill={isActive ? `${color}22` : "#0f172a"}
                    stroke={color}
                    strokeWidth={isActive ? 2 : 0.7}
                    strokeOpacity={isActive ? 1 : 0.2}
                    filter={isActive ? "url(#glow-active)" : undefined}
                    initial={{ opacity: 0.3 }}
                    animate={{ opacity: isActive ? 1 : 0.4 }}
                    transition={{ duration: 0.3 }}
                  />
                  {/* MLP layer 1: Linear (up-project) */}
                  <rect x={ex + 3} y={ey + 4} width={EXPERT_W - 6} height={8}
                    rx={2} fill={isActive ? color : "#1e293b"} opacity={isActive ? 0.7 : 0.15} />
                  {/* GELU label */}
                  <text x={ex + EXPERT_W / 2} y={ey + 21} textAnchor="middle"
                    fill={isActive ? "#e2e8f0" : "#475569"} fontSize={6}>
                    GELU
                  </text>
                  {/* MLP layer 2: Linear (down-project) */}
                  <rect x={ex + 3} y={ey + 25} width={EXPERT_W - 6} height={8}
                    rx={2} fill={isActive ? color : "#1e293b"} opacity={isActive ? 0.7 : 0.15} />
                  {/* Expert label */}
                  <text x={ex + EXPERT_W / 2} y={ey + EXPERT_H - 2} textAnchor="middle"
                    fill={isActive ? "#f1f5f9" : "#475569"} fontSize={7} fontWeight={isActive ? 700 : 400}>
                    E{ei}
                  </text>
                </g>
              );
            })}

            {/* ── Lines from experts to weighted sum ─ */}
            {Array.from({ length: nExperts }, (_, ei) => {
              const col = ei % expertCols;
              const row = Math.floor(ei / expertCols);
              const ex = expertsStartX + col * (EXPERT_W + EXPERT_GAP) + EXPERT_W / 2;
              const ey = expertsY + row * (EXPERT_H + EXPERT_GAP) + EXPERT_H;
              const isActive = active.has(ei);
              const color = EXPERT_COLORS[ei % EXPERT_COLORS.length];

              return isActive ? (
                <motion.line key={`sum-${li}-${ei}`}
                  x1={ex} y1={ey} x2={centerX} y2={sumY - SUM_R}
                  stroke={color} strokeWidth={1.5} strokeOpacity={0.6}
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  transition={{ duration: 0.3, delay: 0.15 }}
                />
              ) : null;
            })}

            {/* ── Weighted Sum ──────────────────── */}
            <circle cx={centerX} cy={sumY} r={SUM_R}
              fill="#1e1b4b" stroke="#7c3aed" strokeWidth={1.2} />
            <text x={centerX} y={sumY + 4} textAnchor="middle" fill="#c4b5fd" fontSize={10} fontWeight={700}>
              Σ
            </text>

            {/* Residual skip arrow (MoE) */}
            <line x1={bx + BLOCK_W - 10} y1={routerY - 4} x2={bx + BLOCK_W - 10} y2={sumY + SUM_R + 4}
              stroke="#334155" strokeWidth={1} strokeDasharray="3 2" />
            <text x={bx + BLOCK_W - 7} y={sumY + SUM_R + 12} textAnchor="middle" fill="#475569" fontSize={8}>+</text>
          </g>
        );
      })}
    </svg>
  );
}
