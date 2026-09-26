"""
graph.py
--------
MAIN ORCHESTRATOR — LangGraph Multi-Agent Supervisor

Ye file sab 5 agents ko ek graph mein jodti hai. Pattern "Supervisor" hai:
- Ek "supervisor" node hai jo LLM se decide karwata hai ke abhi konsa agent
  chalna chahiye (user ki query ko dekh kar).
- Har agent apna kaam kar ke wapas supervisor ke paas aata hai.
- Supervisor jab "FINISH" bole to graph khatam ho kar final_answer bana deta hai.

Use hoti hai: app.py (Streamlit) se — app.py sirf build_graph() ko call
karta hai aur graph.invoke(state) se pura multi-agent system chala deta hai.
"""

import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from state import AgentState
from llm_utils import pick_core_llm
from research_agent import run_research_agent
from analysis_agent import run_analysis_agent
from listing_agent import run_listing_agent
from qa_agent import run_qa_agent
from advertising_agent import run_advertising_agent
from security import check_scope, REFUSAL_MESSAGE, build_guarded_prompt, sanitize_user_input

AGENTS = ["research", "analysis", "listing", "qa", "advertising"]

_BASE_SUPERVISOR_PROMPT = """Aap ek Amazon Seller Multi-Agent System ke Supervisor hain.
Aapke paas 5 specialist agents hain:

1. research     -> Keepa/Helium10 se naye products dhoondhta hai (demand, competition)
2. analysis     -> research data ko pandas code se analyze karta hai (profit margin, scores)
3. listing      -> Amazon listing likhta hai (title, bullets, description, keywords)
4. qa           -> buyer ke sawaalon ka jawab deta hai (listing/analysis context se)
5. advertising  -> multiple LLMs (OpenAI/Grok/Gemini) se ad copy banata hai

User ki request aur ab tak jo kaam ho chuka hai use dekh kar decide karein
ke AAGE konsa agent chalana hai. Agar user ki request ka jawab poora ho
chuka hai (jo agents chalne chahiye the chal chuke), to "FINISH" bolein.

Rules:
- Naya product dhoondhna ho -> research (agar research_data khali hai)
- Research ke baad numbers/profit dekhna ho -> analysis
- Listing/title/bullets chahiye ho -> listing (agar analysis ho chuki ho to behtar)
- Customer ka sawaal ho product ke baare mein -> qa
- Ad copy / marketing text chahiye ho -> advertising (listing ke baad)
- Agar user ne pura pipeline maanga ho (research se ad tak), to sab ek ek
  kar ke chalayein, aakhir mein FINISH.

SIRF ek lafz mein jawab dein: research | analysis | listing | qa | advertising | FINISH
"""

SUPERVISOR_PROMPT = build_guarded_prompt(_BASE_SUPERVISOR_PROMPT)


def guard_node(state: AgentState) -> AgentState:
    """
    Graph ka SABSE PEHLA node. security.py ke rules ke mutabiq check karta
    hai ke (a) query prompt-injection to nahi, (b) query is system ke
    scope (Amazon product research/listing/ads) mein hai ya nahi.
    Agar fail ho, seedha refusal de kar graph khatam kar deta hai — koi
    agent nahi chalta, koi API key waste nahi hoti.
    """
    llm, provider = pick_core_llm(state["api_keys"])
    in_scope, reason = check_scope(llm, state["user_query"])

    state["next_agent"] = "supervisor" if in_scope else "refuse"
    if not in_scope:
        state["final_answer"] = REFUSAL_MESSAGE
    return state


def route_after_guard(state: AgentState) -> str:
    return state.get("next_agent", "refuse")


def supervisor_node(state: AgentState) -> AgentState:
    llm, provider = pick_core_llm(state["api_keys"])

    progress = {
        "research_done": bool(state.get("research_data")),
        "analysis_done": bool(state.get("analysis_result")),
        "listing_done": bool(state.get("listing_content")),
        "qa_done": bool(state.get("qa_response")),
        "advertising_done": bool(state.get("ad_copy")),
    }

    resp = llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=sanitize_user_input(
            f"User request: {state['user_query']}\n"
            f"Progress so far: {json.dumps(progress)}\n"
            "Agla agent kaunsa chalega?"
        )),
    ])

    decision = resp.content.strip().lower()
    decision = next((a for a in AGENTS if a in decision), "finish")

    state["next_agent"] = decision
    return state


def finalize_node(state: AgentState) -> AgentState:
    """Supervisor ne FINISH bola — sab agents ke results ko ek final answer mein combine karta hai."""
    llm, provider = pick_core_llm(state["api_keys"])

    summary_context = {
        "research_data": state.get("research_data"),
        "analysis_result": state.get("analysis_result"),
        "listing_content": state.get("listing_content"),
        "qa_response": state.get("qa_response"),
        "ad_copy": state.get("ad_copy"),
    }

    resp = llm.invoke([
        SystemMessage(content=(
            "Neeche multi-agent system ke sab results hain. User ke liye ek "
            "saaf, mukhtasar Urdu-English mix summary likhein — jo bhi agents "
            "chale unke key results highlight karein."
        )),
        HumanMessage(content=json.dumps(summary_context, ensure_ascii=False)),
    ])

    state["final_answer"] = resp.content.strip()
    return state


def route_after_supervisor(state: AgentState) -> str:
    decision = state.get("next_agent", "finish")
    if decision == "finish":
        return "finalize"
    return decision


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("guard", guard_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", run_research_agent)
    graph.add_node("analysis", run_analysis_agent)
    graph.add_node("listing", run_listing_agent)
    graph.add_node("qa", run_qa_agent)
    graph.add_node("advertising", run_advertising_agent)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("guard")

    graph.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            "supervisor": "supervisor",
            "refuse": END,
        },
    )

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "research": "research",
            "analysis": "analysis",
            "listing": "listing",
            "qa": "qa",
            "advertising": "advertising",
            "finalize": "finalize",
        },
    )

    # Har agent apna kaam kar ke wapas supervisor ke paas jaata hai
    for agent_name in AGENTS:
        graph.add_edge(agent_name, "supervisor")

    graph.add_edge("finalize", END)

    return graph.compile()
