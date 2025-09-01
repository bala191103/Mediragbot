# llm_as_judge - calculating the scores


import time
import json
import re

def judge_score(client, model, question, answer, contexts, dimension: str, llm_chat_fn):
    ctx_joined = "\n\n".join([c.page_content for c in contexts]) if contexts else "NO_CONTEXT"
    rubric = {
        "faithfulness": "Score 1.0 if the answer is fully supported by the context, 0.0 if it contradicts or invents facts; scale smoothly otherwise.",
        "answer_relevancy": "Score 1.0 if the answer fully addresses the user's question, 0.0 if irrelevant; scale if partial.",
        "context_relevancy": "Score 1.0 if the retrieved context is relevant and useful to answer the question, 0.0 if irrelevant; scale if partially relevant."
    }[dimension]

    prompt = f"""
You are an impartial evaluator for a Retrieval-Augmented Generation system.

Dimension to score: {dimension}
Rubric: {rubric}

Question:
{question}

Answer:
{answer}

Retrieved Context:
{ctx_joined}

Return ONLY a JSON object like:
{{"score": 0.0, "reason": "brief reason"}}
"""
    try:
        out = llm_chat_fn(client, model, prompt, temperature=0.0, max_tokens=200)
        m = re.search(r"\{.*\}", out, flags=re.DOTALL)
        data = json.loads(m.group(0)) if m else {"score": None, "reason": "parse_error"}
        score = data.get("score", None)
        if isinstance(score, (int, float)):
            score = max(0.0, min(1.0, float(score)))
        else:
            score = None
        return score, data.get("reason", "")
    except Exception as e:
        return None, f"judge_error: {e}"


def evaluate_realtime(client, model, question, answer, contexts, llm_chat_fn):
    start = time.time()
    f_score, f_reason = judge_score(client, model, question, answer, contexts, "faithfulness", llm_chat_fn)
    a_score, a_reason = judge_score(client, model, question, answer, contexts, "answer_relevancy", llm_chat_fn)
    c_score, c_reason = judge_score(client, model, question, answer, contexts, "context_relevancy", llm_chat_fn)
    latency = time.time() - start
    return {
        "faithfulness": f_score,
        "answer_relevancy": a_score,
        "context_relevancy": c_score,
        "latency_sec": latency,
        "reasons": {
            "faithfulness": f_reason,
            "answer_relevancy": a_reason,
            "context_relevancy": c_reason
        }
    }
