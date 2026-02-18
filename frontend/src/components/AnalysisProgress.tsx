import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { fetchSamples } from "../lib/api";

interface Props {
  readonly step: number; // 0, 1, or 2 matching the AnalyzerView steps
}

export default function AnalysisProgress({ step }: Props) {
  const [samples, setSamples] = useState<string[]>([]);
  const [currentSample, setCurrentSample] = useState<string>("");
  const [llmMessageIndex, setLlmMessageIndex] = useState(0);
  
  // Stable random values for visualizations using lazy state initialization
  const [streamLines] = useState(() => Array.from({ length: 3 }).map((_, i) => ({
    id: i,
    width: 40 + Math.random() * 40,
    delay: i * 0.2
  })));

  const [expertNodes] = useState(() => Array.from({ length: 4 }).map((_, i) => ({
    id: i,
    delay: Math.random()
  })));

  // Fetch samples once
  useEffect(() => {
    fetchSamples().then((data) => {
      // Shuffle samples for variety
      setSamples([...data].sort(() => Math.random() - 0.5));
    }).catch(() => {
      setSamples(["Loading sample texts...", "Analyzing language patterns...", "Processing expert routing..."]);
    });
  }, []);

  // Cycle samples effect for Step 0 and 1
  useEffect(() => {
    if (samples.length === 0) return;
    
    const interval = setInterval(() => {
      setCurrentSample(samples[Math.floor(Math.random() * samples.length)]);
    }, 150);
    
    return () => clearInterval(interval);
  }, [samples, step]);

  // Cycle LLM messages for Step 2
  useEffect(() => {
    if (step !== 2) return;
    const interval = setInterval(() => {
      setLlmMessageIndex(prev => (prev + 1) % 3);
    }, 2000);
    return () => clearInterval(interval);
  }, [step]);

  const llmMessages = [
    "Generating expert labels...",
    "identifying specializations...",
    "Synthesizing descriptions..."
  ];

  return (
    <div className="w-full h-48 bg-slate-950/50 rounded-xl border border-slate-800/50 relative overflow-hidden flex flex-col items-center justify-center p-6">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-indigo-500/5 via-transparent to-transparent" />
      
      {/* Step 0: Feed samples */}
      {step === 0 && (
        <div className="relative w-full max-w-lg text-center z-10">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-slate-500 text-xs font-mono mb-2 uppercase tracking-widest"
          >
            Ingesting Corpus
          </motion.div>
          <div className="h-16 flex items-center justify-center">
            <AnimatePresence mode="wait">
              <motion.div
                key={currentSample}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="text-slate-200 font-medium text-lg leading-snug"
              >
                "{currentSample}"
              </motion.div>
            </AnimatePresence>
          </div>
          <div className="mt-4 flex justify-center gap-1">
            {Array.from({ length: 5 }).map((_, i) => (
              <motion.div
                key={`loading-dot-${i}`}
                className="w-1.5 h-1.5 rounded-full bg-indigo-500"
                animate={{ opacity: [0.2, 1, 0.2] }}
                transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.15 }}
              />
            ))}
          </div>
        </div>
      )}

      {/* Step 1: Record Routing */}
      {step === 1 && (
        <div className="relative w-full z-10 flex flex-col items-center">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-slate-500 text-xs font-mono mb-6 uppercase tracking-widest"
          >
            Mapping Expert Activations
          </motion.div>
          
          <div className="flex gap-8 items-center justify-center w-full">
            {/* Abstract Token Stream */}
            <div className="flex flex-col gap-2 items-end opacity-60">
              {streamLines.map((line) => (
                <motion.div
                  key={line.id}
                  className="h-2 rounded-full bg-slate-600"
                  style={{ width: line.width }}
                  animate={{ x: [0, 20, 0], opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 1.5, repeat: Infinity, delay: line.delay }}
                />
              ))}
            </div>

            {/* Router / Gate */}
            <div className="relative">
              <div className="w-12 h-12 rounded-lg border-2 border-indigo-500/50 bg-indigo-900/20 flex items-center justify-center">
                <span className="text-xl">⚡</span>
              </div>
              <motion.div
                className="absolute inset-0 rounded-lg border border-indigo-400"
                animate={{ scale: [1, 1.2], opacity: [1, 0] }}
                transition={{ duration: 1, repeat: Infinity }}
              />
            </div>

            {/* Expert Buckets */}
            <div className="grid grid-cols-2 gap-2">
              {expertNodes.map((node) => (
                <motion.div
                  key={node.id}
                  className="w-8 h-8 rounded bg-slate-800/80 border border-slate-700 flex items-center justify-center text-[10px] text-slate-500"
                  animate={{ 
                    borderColor: ["#334155", "#6366f1", "#334155"],
                    backgroundColor: ["#1e293b", "#312e81", "#1e293b"]
                  }}
                  transition={{ duration: 0.5, repeat: Infinity, delay: node.delay }}
                >
                  E{node.id}
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Step 2: LLM Analysis */}
      {step === 2 && (
        <div className="relative w-full z-10 flex flex-col items-center">
           <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-slate-500 text-xs font-mono mb-4 uppercase tracking-widest"
          >
            LLM Profiling
          </motion.div>

          <div className="relative">
            <motion.div
              className="text-4xl mb-4"
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            >
              🧠
            </motion.div>
            <motion.div
              className="absolute -top-2 -right-2 text-xl"
              animate={{ opacity: [0, 1, 0], y: -10 }}
              transition={{ duration: 1.5, repeat: Infinity }}
            >
              ✨
            </motion.div>
          </div>

          <div className="h-6 overflow-hidden relative">
             <AnimatePresence mode="wait">
               <motion.div
                 key={llmMessageIndex}
                 initial={{ y: 20, opacity: 0 }}
                 animate={{ y: 0, opacity: 1 }}
                 exit={{ y: -20, opacity: 0 }}
                 transition={{ duration: 0.3 }}
                 className="text-indigo-400 font-medium"
               >
                 {llmMessages[llmMessageIndex]}
               </motion.div>
             </AnimatePresence>
          </div>
          
          <p className="text-slate-500 text-xs mt-2 max-w-xs text-center">
            Identifying specialization patterns and generating human-readable descriptions...
          </p>
        </div>
      )}
    </div>
  );
}
