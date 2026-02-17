export interface ModelInfo {
  n_layer: number;
  n_expert: number;
  top_k: number;
  n_embd: number;
  n_head: number;
  expert_dim: number;
  vocab_size: number;
  block_size: number;
  n_params: number;
}

export interface LayerInfo {
  router_probs: number[][];   // [T, n_expert]
  topk_indices: number[][];   // [T, top_k]
  topk_probs: number[][];     // [T, top_k]
}

export interface InferResponse {
  input_tokens: string[];
  generated_tokens: string[];
  layers: LayerInfo[];
  generated_layers: LayerInfo[][];
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch('/api/model-info');
  if (!res.ok) throw new Error('Failed to fetch model info');
  return res.json();
}

export async function runInference(
  text: string,
  maxNewTokens = 64,
  temperature = 0.8,
): Promise<InferResponse> {
  const res = await fetch('/api/infer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      max_new_tokens: maxNewTokens,
      temperature,
    }),
  });
  if (!res.ok) throw new Error('Inference failed');
  return res.json();
}
