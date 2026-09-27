export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include', body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data as T;
}

export type ModelPrediction = {
  model: string;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  gradcamUrl: string | null;
  limeUrl: string | null;
};

export type PredictionResponse = {
  modality: string;
  model: string;
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  models?: ModelPrediction[];
  imageUrl?: string;
  limeUrl?: string | null;
  shapUrl?: string | null;
  shapDescription?: string | null;
  input?: Record<string, string>;
  clinicalInsights: {
    title: string;
    analysis: string;
    key_findings: string[];
    recommendations: string[];
    disclaimer: string;
    severity: string;
  };
};

export async function predictImage(form: FormData): Promise<PredictionResponse> {
  const response = await fetch('/api/predict/image', { method: 'POST', credentials: 'include', body: form });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Prediction failed (${response.status})`);
  return data as PredictionResponse;
}

export async function predictClinical(input: Record<string, string>): Promise<PredictionResponse> {
  return postJson<PredictionResponse>('/api/predict/clinical', input);
}

export type CsvCohortResponse = {
  row_count: number;
  explained_rows: number;
  shap_url: string;
  chart_description: string;
  lime_url: string | null;
  lime_description: string | null;
  predictions: { row: number; prediction: string; probability_yes: number }[];
  class_counts: { YES: number; NO: number };
};

export async function predictCsv(file: File): Promise<CsvCohortResponse> {
  const form = new FormData();
  form.set('file', file);
  const response = await fetch('/api/predict/csv', { method: 'POST', credentials: 'include', body: form });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `CSV analysis failed (${response.status})`);
  return data as CsvCohortResponse;
}

export type ChatResponse = {
  answer: string;
  sources: { title: string; url: string }[];
  conversation_id: string;
};

export async function sendChat(message: string, conversationId: string, context?: Record<string, unknown>, history: { role: 'user' | 'assistant'; content: string }[] = []): Promise<ChatResponse> {
  return postJson<ChatResponse>('/api/chat', { message, conversation_id: conversationId, context, history });
}

export async function downloadReport(body: unknown) {
  const response = await fetch('/download_report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include', body: JSON.stringify(body) });
  if (!response.ok) { const data = await response.json(); throw new Error(data.error || 'Report generation failed'); }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a'); link.href = url; link.download = 'Multimodal_Lung_Cancer_Research_Report.pdf'; link.click(); URL.revokeObjectURL(url);
}
