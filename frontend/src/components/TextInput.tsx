import { useState } from "react";

interface Props {
  readonly onSubmit: (text: string) => void;
  readonly loading: boolean;
}

const EXAMPLES = [
  "Once upon a time",
  "The little cat was very",
  "She wanted to play with",
  "One day, a big bear",
];

export default function TextInput({ onSubmit, loading }: Props) {
  const [text, setText] = useState("");

  const handleSubmit = () => {
    const input = text.trim();
    if (input) onSubmit(input);
  };

  return (
    <div className="space-y-3">
      <div className="relative">
        <textarea
          className="w-full bg-slate-900/80 border border-slate-700/60 rounded-lg px-4 py-3 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/30 resize-none font-mono transition-colors"
          rows={3}
          placeholder="Type text to see how the MoE routes tokens to experts..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit();
            }
          }}
          disabled={loading}
        />
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={handleSubmit}
          disabled={loading || !text.trim()}
          className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm font-medium rounded-lg transition-colors cursor-pointer disabled:cursor-not-allowed"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Running...
            </span>
          ) : (
            "Run Inference"
          )}
        </button>

        <div className="flex gap-1.5 ml-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => { setText(ex); onSubmit(ex); }}
              disabled={loading}
              className="px-2.5 py-1 text-[11px] text-slate-400 bg-slate-800/60 hover:bg-slate-700/60 hover:text-slate-300 rounded-md border border-slate-700/40 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed truncate max-w-[140px]"
              title={ex}
            >
              {ex}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
