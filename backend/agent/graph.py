"""Graph construction helper for the agent.

This module exposes a single async function `build_graph` which initializes
the model and tools and returns a compiled LangGraph `StateGraph` ready
to be served by the API. The function is intended to be called during the
FastAPI startup event so that networked initialization happens at startup
and not at import time.
"""

import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.model import create_model
from agent.nodes.chat import ChatNode
from agent.routing import custom_tools_condition
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
    - Discover tools asynchronously via `get_tools`.
    - Construct the StateGraph with a ReAct loop.
    - Compile the graph with an in-memory checkpointer.

    Returns:
            A compiled `StateGraph` instance ready for execution.
    """

    # Load langsmith API key from file only if set in environment
    load_langsmith_api_key_from_file()

    model = create_model("google_genai:gemini-2.5-flash")
    tools = await get_tools()

    # --- ReAct graph with chat and tools ---
    react_graph = StateGraph(OverallState)

    react_graph.add_node("chat", ChatNode(model, tools))
    react_graph.add_node("tools", ToolNode(tools))

    react_graph.add_edge(START, "chat")
    react_graph.add_conditional_edges("chat", custom_tools_condition)
    react_graph.add_edge("tools", "chat")
    react_graph.add_edge("chat", END)

    compiled_react = react_graph.compile(checkpointer=MemorySaver())

    # Visualize the graph for debugging
    print(compiled_react.get_graph().draw_mermaid())

    return compiled_react
