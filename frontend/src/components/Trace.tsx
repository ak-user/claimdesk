import type { ToolStep } from "../types";

export function Trace({ steps, provider }: { steps: ToolStep[]; provider: string }) {
  if (steps.length === 0) return null;
  const total = steps.reduce((sum, step) => sum + step.ms, 0);
  return (
    <details className="trace">
      <summary>
        Agent trace · {steps.length} tool {steps.length === 1 ? "call" : "calls"} ·{" "}
        {total.toFixed(1)} ms · {provider}
      </summary>
      <ol>
        {steps.map((step, index) => (
          <li key={index}>
            <code className="tool-name">{step.tool}</code>
            <code className="tool-args">{JSON.stringify(step.arguments)}</code>
            <span className="tool-summary">{step.summary}</span>
            <span className="tool-ms">{step.ms.toFixed(1)} ms</span>
          </li>
        ))}
      </ol>
    </details>
  );
}
