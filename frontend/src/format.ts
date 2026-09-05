const currency = new Intl.NumberFormat("en-CA", { style: "currency", currency: "CAD" });
const longDate = new Intl.DateTimeFormat("en-CA", {
  year: "numeric",
  month: "short",
  day: "numeric",
});

export const money = (value: number) => currency.format(value);

export function formatDate(iso: string): string {
  const [year, month, day] = iso.split("-").map(Number);
  if (!year || !month || !day) return iso;
  return longDate.format(new Date(year, month - 1, day));
}

const FLAG_TEXT: Record<string, string> = {
  answer_has_no_citation: "The answer doesn't cite a section.",
  model_unavailable_answered_offline: "The model didn't respond, so the offline planner answered.",
  step_limit_reached: "The agent stopped at its step limit.",
  possible_prompt_injection: "The document tried to give instructions to the AI.",
  model_extraction_failed_used_rules: "Model extraction failed, so the rule-based reader was used.",
};

export function describeFlag(flag: string): string {
  const [code, detail] = flag.split(":");
  if (code === "removed_unsupported_citation") {
    return `Removed a citation to section ${detail}, which no tool returned.`;
  }
  return FLAG_TEXT[code] ?? flag;
}
