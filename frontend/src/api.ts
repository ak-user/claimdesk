import type {
  Answer,
  ClaimAssessment,
  Health,
  QueuedClaim,
  ReviewDecision,
  ReviewStatus,
  Sample,
  Turn,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new Error("Can't reach the ClaimDesk API. Is the backend running on port 8000?");
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) {
        detail = body.detail.map((item: { msg: string }) => item.msg).join("; ");
      }
    } catch {
      // Keep the status text.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/api/health"),
  ask: (question: string, memberId: string | null, history: Turn[] = []) =>
    request<Answer>("/api/ask", json({ question, member_id: memberId, history })),
  assess: (text: string) => request<ClaimAssessment>("/api/claims/assess", json({ text })),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<ClaimAssessment>("/api/claims/upload", { method: "POST", body: form });
  },
  submit: (text: string) => request<QueuedClaim>("/api/claims", json({ text })),
  queue: (status: ReviewStatus | null) =>
    request<QueuedClaim[]>(status ? `/api/claims?status=${status}` : "/api/claims"),
  review: (id: number, decision: ReviewDecision) =>
    request<QueuedClaim>(`/api/claims/${id}/review`, json(decision)),
  samples: () => request<Sample[]>("/api/samples"),
  sample: (name: string) =>
    request<{ name: string; text: string }>(`/api/samples/${encodeURIComponent(name)}`),
};
