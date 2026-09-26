"""
qa_agent.py
-----------
AGENT 4 — CUSTOMER Q&A AGENT

Kaam: Buyer/customer product ke baare mein jo bhi sawaal puchein, unka
jawab dena — listing_content, analysis_result, aur research_data ko
context ki tarah use karte hue (taake jawab accurate aur product-specific ho).

Use hoti hai: graph.py se — supervisor route="qa" karta hai to
graph.py isi file ke run_qa_agent() ko call karta hai.
"""

import json
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from llm_utils import pick_core_llm
from security import build_guarded_prompt, sanitize_user_input, REFUSAL_MESSAGE

_BASE_QA_PROMPT = f"""Aap ek Amazon Customer Service Assistant hain.
Neeche product ki poori context (research + analysis + listing) di gayi hai.
Customer ka sawaal isi context ke hisaab se answer karein — helpful,
short, aur honest.

Zaroori: Agar context mein is sawaal ka jawab maujood na ho, to guess ya
andaza mat lagayein — saaf keh dein: "Sorry, mere paas iska exact data
nahi hai." Agar sawaal product se related hi na ho, to "{REFUSAL_MESSAGE}"
keh dein.
"""

QA_SYSTEM_PROMPT = build_guarded_prompt(_BASE_QA_PROMPT)


def run_qa_agent(state: AgentState) -> AgentState:
    llm, provider = pick_core_llm(state["api_keys"])

    context = {
        "listing_content": state.get("listing_content"),
        "analysis_result": state.get("analysis_result"),
        "research_data": state.get("research_data"),
    }

    resp = llm.invoke([
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(content=sanitize_user_input(
            f"Product context:\n{json.dumps(context, ensure_ascii=False)}\n\n"
            f"Customer ka sawaal: {state['user_query']}"
        )),
    ])

    answer = resp.content.strip()

    state["qa_response"] = answer
    state["messages"] = state.get("messages", []) + [
        HumanMessage(content=f"[QA Agent Result]\n{answer}")
    ]
    state["next_agent"] = "supervisor"
    return state
