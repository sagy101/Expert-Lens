import type { ModelInfo } from "../lib/api";

interface Props {
  readonly modelInfo: ModelInfo | null;
}

export default function InfoPanel({ modelInfo }: Props) {
  return (
    <div className="text-xs text-slate-400 space-y-3">
      <div>
        <h3 className="text-indigo-400 font-semibold text-sm mb-1">What is MoE?</h3>
        <p className="leading-relaxed">
          In a standard transformer, every token passes through the <em>same</em> feed-forward
          network (FFN). A <strong className="text-slate-300">Mixture of Experts</strong> replaces
          that single FFN with <strong className="text-slate-300">multiple smaller expert networks</strong> and
          a <strong className="text-slate-300">learned router</strong> that picks which experts to
          activate for each token. Only a few experts fire per token — making the model
          much larger in total capacity while keeping compute constant.
        </p>
      </div>

      <div>
        <h3 className="text-indigo-400 font-semibold text-sm mb-1">This Model</h3>
        {modelInfo ? (
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            <Stat label="Parameters" value={`${(modelInfo.n_params / 1e6).toFixed(1)}M`} />
            <Stat label="Layers" value={modelInfo.n_layer} />
            <Stat label="Experts / layer" value={modelInfo.n_expert} />
            <Stat label="Active / token" value={`top-${modelInfo.top_k}`} />
            <Stat label="Embed dim" value={modelInfo.n_embd} />
            <Stat label="Attention heads" value={modelInfo.n_head} />
            <Stat label="Expert hidden" value={modelInfo.expert_dim} />
            <Stat label="Vocab size" value={modelInfo.vocab_size} />
          </div>
        ) : (
          <p className="text-slate-500">Loading model info...</p>
        )}
      </div>

      <div>
        <h3 className="text-indigo-400 font-semibold text-sm mb-1">How to Read</h3>
        <ul className="space-y-1 leading-relaxed list-none">
          <li><strong className="text-slate-300">Network diagram</strong> — glowing nodes = active experts for the selected token</li>
          <li><strong className="text-slate-300">Heatmap</strong> — router probability for every token × expert combination</li>
          <li><strong className="text-slate-300">Usage bars</strong> — how evenly the router distributes tokens across experts</li>
        </ul>
      </div>
    </div>
  );
}

function Stat({ label, value }: { readonly label: string; readonly value: string | number }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-500">{label}</span>
      <span className="text-slate-300 font-mono">{value}</span>
    </div>
  );
}
