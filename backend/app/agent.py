import os
from typing import Annotated
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.graph import START, END, StateGraph
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from copilotkit import CopilotKitState

AI_API_KEY_FILE = os.getenv("AI_API_KEY_FILE")
with open(AI_API_KEY_FILE, "r") as f:
    AI_API_KEY = f.read().strip()
ELASTICSEARCH_MCP_ENDPOINT = os.getenv("ELASTICSEARCH_MCP_ENDPOINT")
MAPS_MCP_ENDPOINT = os.getenv("MAPS_MCP_ENDPOINT")

class State(CopilotKitState):
    messages: Annotated[list[AnyMessage], add_messages]

async def build_graph():
    model = init_chat_model("google_genai:gemini-2.5-flash", api_key = AI_API_KEY)

    client = MultiServerMCPClient(
        {
            "elasticsearch": {
                "url": ELASTICSEARCH_MCP_ENDPOINT,
                "transport": "streamable_http",
            },
            "maps": {
                "url": MAPS_MCP_ENDPOINT,
                "transport": "streamable_http",
            }
        }
    )

    tools = await client.get_tools()

    def chat_node(state: State) -> State:
        state['messages'] = model.bind_tools(tools).invoke(state['messages'])
        return state

    grap_builder = StateGraph(State)
    grap_builder.add_node(chat_node)
    grap_builder.add_node(ToolNode(tools))
    grap_builder.add_edge(START, "chat_node")
    grap_builder.add_conditional_edges(
        "chat_node",
        tools_condition,
    )
    grap_builder.add_edge("tools", "chat_node")
    grap_builder.add_edge("chat_node", END)
    return grap_builder.compile(checkpointer = MemorySaver())