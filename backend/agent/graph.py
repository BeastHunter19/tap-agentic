"""Graph construction helper for the agent.

This module exposes a single async function `build_graph` which initializes
the model and MCP tools and returns a compiled LangGraph `StateGraph` ready
to be served by the API. The function is intended to be called during the
FastAPI startup event so that networked initialization happens at startup
and not at import time.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.chat_node import ChatNode
from agent.model import create_model
from agent.state import OverallState
from agent.tools import get_tools


async def build_graph() -> StateGraph:
    """Create and compile the agent StateGraph.

    This function performs the following steps:
    - Instantiate the chat model via `create_model`.
    - Discover MCP tools asynchronously via `get_tools`.
    - Construct the StateGraph with a Chat node and a ToolNode and wire
        the edges.
    - Compile the graph with an in-memory checkpointer.

    Returns:
            A compiled `StateGraph` instance ready for execution.
    """
    model = create_model()
    tools = await get_tools()

    graph_builder = StateGraph(OverallState)
    graph_builder.add_node("chat", ChatNode(model, tools))
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_edge(START, "chat")
    graph_builder.add_conditional_edges("chat", tools_condition)
    graph_builder.add_edge("tools", "chat")
    graph_builder.add_edge("chat", END)

    return graph_builder.compile(checkpointer=MemorySaver())
