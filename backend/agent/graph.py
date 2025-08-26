"""Graph construction helper for the agent.

This module exposes a single async function `build_graph` which initializes
the model and MCP tools and returns a compiled LangGraph `StateGraph` ready
to be served by the API. The function is intended to be called during the
FastAPI startup event so that networked initialization happens at startup
and not at import time.
"""

import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.chat_node import ChatNode
from agent.model import create_model
from agent.state import OverallState
from agent.tools import get_tools


def load_langsmith_api_key_from_file() -> None:
    """Load the LangSmith API key from a file if the environment variable is set.

    This function checks for the presence of the `LANGSMITH_API_KEY_FILE`
    environment variable. If it is set, it reads the API key from the specified
    file and sets the `LANGSMITH_API_KEY` environment variable accordingly.

    If the `LANGSMITH_API_KEY_FILE` variable is not set, this function does nothing.
    """
    api_key_file = os.getenv("LANGSMITH_API_KEY_FILE")
    if api_key_file and os.path.isfile(api_key_file):
        with open(api_key_file, "r") as file:
            api_key = file.read().strip()
            os.environ["LANGSMITH_API_KEY"] = api_key


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

    # Load langsmith API key from file only if set in environment
    load_langsmith_api_key_from_file()

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
