"""
research_agent.py
------------------
AGENT 1 — RESEARCH AGENT

Kaam: User ke diye gaye niche/keyword se Amazon par sellable products
dhoondhna. Ye agent Keepa aur Helium10 tools ko khud decide kar ke call
karta hai (ReAct pattern: LLM sochta hai "mujhe konsa tool chahiye",
tool call karta hai, result dekhta hai, phir final data return karta hai).

Use hoti hai: graph.py se — supervisor jab route="research" bhejta hai
to graph.py isi file ke run_research_agent() function ko call karta hai.
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from tools import ALL_RESEARCH_TOOLS
from llm_utils import pick_core_llm
from security import build_guarded_prompt, sanitize_user_input

_BASE_RESEARCH_PROMPT = """Aap ek Amazon Product Research Agent hain.
Aapka kaam hai user ke diye niche/keyword ke liye best sellable products
dhoondhna, Keepa aur Helium10 tools istemal karte hue.

Rules:
1. keepa_search aur helium10_search dono tools try karein (jo bhi key available ho).
2. Har tool call mein user ki di API key hi pass karein — kabhi khud se key mat banayein.
3. Result mein har product ka: ASIN, title, price, sales rank, review count, rating
   zaroor include karein.
4. Agar dono tools se error aaye ya empty result aaye, to saaf bata dein ke
   research nahi ho saki aur wajah batayein — koi bhi product khud se mat banayein.
5. Aakhir mein sirf structured JSON-jaisa summary dein — extra chit-chat na karein.
"""

RESEARCH_SYSTEM_PROMPT = build_guarded_prompt(_BASE_RESEARCH_PROMPT)


def run_research_agent(state: AgentState) -> AgentState:
    llm, provider = pick_core_llm(state["api_keys"])

    agent = create_react_agent(llm, tools=ALL_RESEARCH_TOOLS)

    keepa_key = state["api_keys"].get("keepa", "")
    helium10_key = state["api_keys"].get("helium10", "")

    user_instruction = (
        f"User query: {state['user_query']}\n\n"
        f"Keepa API key: {keepa_key}\n"
        f"Helium10 API key: {helium10_key}\n\n"
        "In keys ko tools ko call karte waqt api_key parameter mein pass karein."
    )

    result = agent.invoke({
        "messages": [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=sanitize_user_input(user_instruction)),
        ]
    })

    final_message = result["messages"][-1].content

    state["research_data"] = [{"raw_summary": final_message}]
    state["messages"] = state.get("messages", []) + [
        HumanMessage(content=f"[Research Agent Result]\n{final_message}")
    ]
    state["next_agent"] = "supervisor"
    return state
