export type ModelType = 'char' | 'bpe';

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
  n_active_params: number;
  model_type: ModelType;
  available_models: ModelType[];
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

export interface CharInfo {
  char: string;
  count: number;
  pct: number;
}

export interface ExampleWord {
  word: string;
  highlights: boolean[];
}

export interface ExpertProfileEntry {
  expert_id: number;
  role: string;
  domain: string;
  description: string;
  total_activations: number;
  top_chars: CharInfo[];
  example_words: ExampleWord[];
}

export interface DemoChar {
  char: string;
  expert: number;
}

export interface ExpertProfileResponse {
  layers: ExpertProfileEntry[][];
  demo_sentence: string;
  demo_layers: DemoChar[][];
  sample_count: number;
}

export async function fetchExpertProfile(modelType: ModelType = 'char'): Promise<ExpertProfileResponse> {
  const res = await fetch(`/api/expert-profile?model_type=${modelType}`);
  if (!res.ok) throw new Error('Failed to fetch expert profile');
  return res.json();
}

export async function fetchModelInfo(modelType: ModelType = 'char'): Promise<ModelInfo> {
  const res = await fetch(`/api/model-info?model_type=${modelType}`);
  if (!res.ok) throw new Error('Failed to fetch model info');
  return res.json();
}

export async function runInference(
  text: string,
  maxNewTokens = 64,
  temperature = 0.8,
  modelType: ModelType = 'char',
): Promise<InferResponse> {
  const res = await fetch('/api/infer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      max_new_tokens: maxNewTokens,
      temperature,
      model_type: modelType,
    }),
  });
  if (!res.ok) throw new Error('Inference failed');
  return res.json();
}
