"""
listing_agent.py
-----------------
AGENT 3 — LISTING AGENT

Kaam: Analysis Agent ke result aur user ke product details se Amazon
listing likhna: SEO-friendly title, 5 bullet points, description,
aur backend search keywords.

Use hoti hai: graph.py se — supervisor route="listing" karta hai to
graph.py isi file ke run_listing_agent() ko call karta hai.
"""

import json
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm_utils import pick_core_llm
from security import build_guarded_prompt, sanitize_user_input

_BASE_LISTING_PROMPT = """Aap ek Amazon Listing Copywriting expert hain.
Diye gaye product/analysis data se ek complete Amazon listing likhein.

Rules:
- Sirf diye gaye research/analysis data par based facts use karein — koi
  feature, certification, ya claim khud se mat banayein jo data mein na ho.
- Agar context mein product data hi na ho, to listing ki jagah bata dein
  ke "listing banane ke liye pehle product research/analysis data chahiye".
- Koi bhi misleading ya false claim (fake certification, jhoota health
  claim) shamil na karein.

Output STRICTLY is JSON format mein dein (koi extra text nahi):
{
  "title": "...",              // max 200 characters, keywords front-loaded
  "bullet_points": ["...", "...", "...", "...", "..."],  // exactly 5, benefit-focused
  "description": "...",        // 1500-2000 characters, HTML-free plain text
  "backend_keywords": "..."    // comma-separated, max 250 bytes, no repeats from title
}
"""

LISTING_SYSTEM_PROMPT = build_guarded_prompt(_BASE_LISTING_PROMPT)


def run_listing_agent(state: AgentState) -> AgentState:
    llm, provider = pick_core_llm(state["api_keys"])

    context = {
        "user_query": state.get("user_query"),
        "research_data": state.get("research_data"),
        "analysis_result": state.get("analysis_result"),
    }

    resp = llm.invoke([
        SystemMessage(content=LISTING_SYSTEM_PROMPT),
        HumanMessage(content=sanitize_user_input(json.dumps(context, ensure_ascii=False))),
    ])

    content = resp.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        listing = json.loads(content)
    except json.JSONDecodeError:
        listing = {"raw_output": content, "note": "JSON parse nahi ho saka, raw text dikhaya ja raha hai."}

    state["listing_content"] = listing
    state["messages"] = state.get("messages", []) + [
        HumanMessage(content=f"[Listing Agent Result]\n{json.dumps(listing, ensure_ascii=False)}")
    ]
    state["next_agent"] = "supervisor"
    return state
