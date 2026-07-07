import json
import time
import pathlib
import mlflow

# ── Config ────────────────────────────────────────────────────────────────────
LOG_FILE = pathlib.Path("llmops_log.jsonl")

# Approximate cost per 1,000 tokens for a local model is $0.00 (no API cost).
# Swap these values out if you later move to a paid provider like OpenAI/Anthropic.
COST_PER_1K_PROMPT     = 0.0      # USD
COST_PER_1K_COMPLETION = 0.0      # USD

mlflow.set_tracking_uri("http://0.0.0.0:5001")

# Connect to your active workspace dashboard (creates it if it doesn't exist)
mlflow.set_experiment("career_transition_advisor")


# ── Token Counter Heuristic ───────────────────────────────────────────────────
def _count_tokens(text: str) -> int:
    """
    A stable character-based heuristic for token estimation on local models.
    LLM tokens average out to roughly ~4 characters per token. 
    Falls back to a minimum of 1 token to prevent division-by-zero errors.
    """
    if not text:
        return 1
    return max(1, len(text) // 4)


def log_request(
    *,
    model: str,
    prompt: str,
    completion: str,
    latency_ms: float,
    guardrail_fired: bool = False,
    extra: dict | None = None,
) -> None:
    """
    Logs one structured JSON line to your local disk and transmits metrics 
    + prompt text to the MLflow dashboard for real-time visualization.
    """
    try:
        # Calculate tokens and estimated cost structures
        prompt_tokens     = _count_tokens(prompt)
        completion_tokens = _count_tokens(completion)
        total_tokens      = prompt_tokens + completion_tokens

        cost_usd = (
            (prompt_tokens     / 1_000) * COST_PER_1K_PROMPT
            + (completion_tokens / 1_000) * COST_PER_1K_COMPLETION
        )

        # Assemble the uniform data record
        record = {
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model":             model,
            "latency_ms":        round(latency_ms, 2),
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      total_tokens,
            "estimated_cost_usd": round(cost_usd, 6),
            "guardrail_fired":   guardrail_fired,
            "raw_prompt":        prompt,
            "raw_completion":    completion,
            **(extra or {}),
        }

        # ── 1. Write the full, untruncated JSON line backup to local disk ──────
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # ── 2. Transmit to your running MLflow Dashboard server ───────────────
        with mlflow.start_run(run_name=f"faq_{int(time.time())}"):
            
            # Safely truncate text to 450 characters to prevent MLflow database crashes
            safe_prompt_snippet = prompt if len(prompt) <= 450 else prompt[:450] + "..."
            safe_completion_snippet = completion if len(completion) <= 450 else completion[:450] + "..."

            # Parameters — Now tracking the text snippets directly in the UI columns
            mlflow.log_params({
                "model":           model,
                "guardrail_fired": str(guardrail_fired),
                "interface":       (extra or {}).get("interface", "api"),
                "user_prompt":     safe_prompt_snippet,
                "bot_response":    safe_completion_snippet,
            })
            
            # Metrics
            mlflow.log_metrics({
                "latency_ms":         round(latency_ms, 2),
                "prompt_tokens":      prompt_tokens,
                "completion_tokens":  completion_tokens,
                "total_tokens":       total_tokens,
                "estimated_cost_usd": round(cost_usd, 6),
            })

    except Exception as exc:
        import sys
        print(f"[llmops_logger] WARNING: Could not log request — {exc}", file=sys.stderr)