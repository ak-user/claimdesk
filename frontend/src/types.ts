// Mirrors backend/claimdesk/models.py.

export type Provider = "offline" | "azure" | "openai";

export interface Health {
  status: string;
  version: string;
  llm_provider: Provider;
  retrieval: "bm25" | "hybrid";
  sections: number;
  as_of: string;
}

export interface Citation {
  section: string;
  title: string;
  snippet: string;
}

export interface ToolStep {
  tool: string;
  arguments: Record<string, unknown>;
  summary: string;
  ms: number;
}

export interface Turn {
  role: "user" | "assistant";
  content: string;
}

export interface Answer {
  answer: string;
  citations: Citation[];
  trace: ToolStep[];
  provider: string;
  refused: boolean;
  flags: string[];
}

export interface ClaimFields {
  member_id: string | null;
  patient_name: string | null;
  service_type: string | null;
  service_date: string | null;
  provider_name: string | null;
  provider_registration: string | null;
  amount: number | null;
}

export interface Issue {
  field: string | null;
  severity: "error" | "warning" | "info";
  kind: "not_payable" | "needs_correction" | "note";
  message: string;
  section: string | null;
}

export interface Estimate {
  benefit: string;
  benefit_display: string;
  coinsurance_percent: number;
  covered_share: number;
  estimate: number;
  remaining_before: number | null;
  remaining_after: number | null;
  capped_by_maximum: boolean;
}

export type ClaimStatus = "ready_for_review" | "needs_correction" | "not_payable";

export interface ClaimAssessment {
  status: ClaimStatus;
  summary: string;
  fields: ClaimFields;
  extractor: "rules" | "llm";
  benefit: string | null;
  benefit_display: string | null;
  issues: Issue[];
  estimate: Estimate | null;
  citations: Citation[];
  flags: string[];
  requires_human_review: boolean;
}

export interface Sample {
  name: string;
  title: string;
}
