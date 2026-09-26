"""
analysis_agent.py
------------------
AGENT 2 — ANALYSIS AGENT

Kaam: Research Agent se aaya hua raw product data lena aur ise CODE
(pandas) se analyze karna — profit margin estimate, competition score,
demand score, aur ek overall "opportunity score" nikalna. Phir LLM se
in numbers ki plain-language explanation likhwana.

Use hoti hai: graph.py se — supervisor route="analysis" karta hai to
graph.py isi file ke run_analysis_agent() ko call karta hai.
"""

import json
import re
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm_utils import pick_core_llm
from security import build_guarded_prompt, sanitize_user_input

EXTRACTION_PROMPT = build_guarded_prompt("""Neeche Research Agent ka raw output diya gaya hai.
Isme se jitne products mil sakein unki list ek JSON array mein nikal kar dein.
Har product object mein ye fields honi chahiye (jo na milein wahan null dein):
asin, title, price (number, USD), sales_rank (number), review_count (number), rating (number).

Agar Research Agent ke output mein koi product data hi nahi hai (ya error/empty
hai), to KHAALI JSON array [] return karein — khud se koi product mat banayein.

SIRF valid JSON array return karein, koi extra text nahi.
""")


def _extract_products(llm, raw_text: str) -> list:
    resp = llm.invoke([
        SystemMessage(content=EXTRACTION_PROMPT),
        HumanMessage(content=sanitize_user_input(raw_text)),
    ])
    content = resp.content.strip()
    # LLM kabhi kabhi ```json fences laga deta hai — unhe hata dein
    content = re.sub(r"^```json|```$", "", content, flags=re.MULTILINE).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return []


def _compute_metrics(products: list, estimated_cost_per_unit: float = None) -> dict:
    """
    Ye asli 'code se analysis' hissa hai — pandas ke saath deterministic
    calculation, LLM ke guess par depend nahi karta.
    """
    if not products:
        return {"error": "Analyze karne ke liye koi product data nahi mila."}

    df = pd.DataFrame(products)

    for col in ["price", "sales_rank", "review_count", "rating"]:
        if col not in df.columns:
            df[col] = None
        df[col] = pd.to_numeric(df[col], errors="coerce")

    avg_price = df["price"].mean(skipna=True)
    avg_reviews = df["review_count"].mean(skipna=True)
    avg_rank = df["sales_rank"].mean(skipna=True)

    # Competition score: zyada reviews + zyada products = zyada competition (0-100, zyada = mushkil)
    competition_score = min(100, (avg_reviews or 0) / 50) if pd.notna(avg_reviews) else None

    # Demand score: behtar (chhota) sales rank = zyada demand (0-100, zyada = behtar)
    demand_score = None
    if pd.notna(avg_rank) and avg_rank > 0:
        demand_score = max(0, 100 - (avg_rank / 1000))

    # Profit margin estimate (agar user ne cost diya ho)
    profit_margin_pct = None
    if estimated_cost_per_unit and pd.notna(avg_price) and avg_price > 0:
        amazon_fees_pct = 0.30  # rough estimate: referral + FBA fees
        est_fees = avg_price * amazon_fees_pct
        profit = avg_price - est_fees - estimated_cost_per_unit
        profit_margin_pct = round((profit / avg_price) * 100, 2)

    opportunity_score = None
    if demand_score is not None and competition_score is not None:
        opportunity_score = round((demand_score * 0.6) + ((100 - competition_score) * 0.4), 2)

    return {
        "product_count": len(df),
        "avg_price_usd": round(avg_price, 2) if pd.notna(avg_price) else None,
        "avg_review_count": round(avg_reviews, 1) if pd.notna(avg_reviews) else None,
        "avg_sales_rank": round(avg_rank, 0) if pd.notna(avg_rank) else None,
        "competition_score_0to100": round(competition_score, 1) if competition_score is not None else None,
        "demand_score_0to100": round(demand_score, 1) if demand_score is not None else None,
        "estimated_profit_margin_pct": profit_margin_pct,
        "opportunity_score_0to100": opportunity_score,
    }


def run_analysis_agent(state: AgentState) -> AgentState:
    llm, provider = pick_core_llm(state["api_keys"])

    raw_research = ""
    if state.get("research_data"):
        raw_research = "\n".join(
            item.get("raw_summary", "") for item in state["research_data"]
        )

    products = _extract_products(llm, raw_research)
    metrics = _compute_metrics(products)

    explain_prompt = (
        "Neeche ek Amazon product research ka numeric analysis hai. "
        "Isko seller ke liye 3-4 line mein aasan Urdu-English mix mein "
        "samjhayein — kya ye product worth pursuing hai ya nahi, aur kyun. "
        "Agar metrics mein 'error' field ho (yani data available hi nahi), "
        "to sirf itna kahein ke data available nahi — koi number khud se mat banayein.\n\n"
        f"{json.dumps(metrics, ensure_ascii=False)}"
    )
    explanation = llm.invoke([
        SystemMessage(content=build_guarded_prompt("Aap ek Amazon data analyst hain.")),
        HumanMessage(content=explain_prompt),
    ]).content

    state["analysis_result"] = {"metrics": metrics, "explanation": explanation}
    state["messages"] = state.get("messages", []) + [
        HumanMessage(content=f"[Analysis Agent Result]\n{json.dumps(metrics)}\n{explanation}")
    ]
    state["next_agent"] = "supervisor"
    return state
