"""VERL reward function — FinQA numeric execution accuracy.

Plugs into verl's reward dispatcher when `data_source == "finqa_exec_acc"`.
Returns 1.0 if the predicted final number matches `ground_truth` within 1e-3
relative tolerance, 0.0 otherwise. Mirrors the answer_normalizer logic used
by run.sh evaluation in our existing FinQA pipeline.

The `solution_str` here is the full trajectory the s3 search agent emitted.
The actual numeric answer is produced by the FROZEN GENERATOR (Bedrock Kimi
K2.5) on the agent's final selected context — the s3 paper's PPO loop calls
it after rollouts terminate via execute_predictions(do_search=False). We
follow the same convention and extract the generator's final answer from
the appended observation block.
"""

import re


def _extract_final_answer(solution_str: str) -> str:
    """Extract the generator's final numeric answer from the trajectory.

    Search order:
    1. Last <answer>...</answer> tag (if present)
    2. Last <information>...</information> block (after do_search=False the
       generator's output is wrapped there by execute_predictions)
    3. Last numeric token in the entire string
    """
    for tag in ("answer", "final_answer"):
        m = list(re.finditer(rf"<{tag}>(.*?)</{tag}>", solution_str, re.DOTALL | re.IGNORECASE))
        if m:
            return m[-1].group(1).strip()
    info = list(re.finditer(r"<information>(.*?)</information>", solution_str, re.DOTALL))
    if info:
        return info[-1].group(1).strip()
    return solution_str.strip()


def _to_float(s: str):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def _numeric_match(prediction_text: str, gold) -> bool:
    if prediction_text is None or gold is None:
        return False
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", prediction_text.replace(",", ""))
    if not nums:
        return False
    g = _to_float(gold)
    if g is None:
        return False
    for n in reversed(nums):  # later numbers more likely to be the answer
        pn = _to_float(n)
        if pn is None:
            continue
        tol = max(1e-3, 1e-3 * abs(g))
        if abs(pn - g) <= tol:
            return True
    return False


def compute_score_finqa(solution_str, ground_truth, method="strict",
                         format_score=0.0, score=1.0):
    """The scoring function for FinQA exec_acc (simple float return).

    Args:
        solution_str: the trajectory text emitted by the s3 search agent
                      (including any final generator output appended by
                      execute_predictions).
        ground_truth: the gold numeric answer (string-castable to float).
        method: ignored (kept for VERL API compatibility).
        format_score: returned when prediction is non-empty but does not match.
        score: returned on exact numeric match.

    Returns:
        float reward in {0.0, format_score, score}.
    """
    pred_text = _extract_final_answer(solution_str)
    if not pred_text:
        return 0.0
    if _numeric_match(pred_text, ground_truth):
        return score
    return format_score


def compute_score_finqa_rag(
    solution_str,
    ground_truth,
    zeroshot_answers=None,
    data_source=None,
    use_utility_score=True,
    use_generation_score=True,
):
    """VERL RewardManager-compatible wrapper.

    Matches rag_2.compute_score_rag signature:
        returns (score, answer_zeroshot, answer_zeroshot_score)
    The latter two are unused for FinQA (we don't compute a generator
    zero-shot baseline at reward time; that's handled by the naive_correct
    precompute embedded in the parquet's data_source filtering).
    """
    s = compute_score_finqa(solution_str, ground_truth)
    return s, None, None
