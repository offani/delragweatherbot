from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.nodes import AgentState, router_node, weather_node, rag_node, generate_node

def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("generate", generate_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Access the state directly in the conditional function
    def route_decision(state):
        return state["source"]

    # Add conditional edges
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "weather": "weather",
            "rag": "rag"
        }
    )

    # Add normal edges
    workflow.add_edge("weather", "generate")
    workflow.add_edge("rag", "generate")
    workflow.add_edge("generate", END)

    # Add checkpointer for conversation memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

graph = build_graph()
