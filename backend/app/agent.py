import os
from typing import Annotated
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.graph import START, END, StateGraph
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from copilotkit import CopilotKitState

GOOGLE_API_KEY_FILE = os.getenv("GOOGLE_API_KEY_FILE")
with open(GOOGLE_API_KEY_FILE, "r") as f:
    GOOGLE_API_KEY = f.read().strip()

class State(CopilotKitState):
    messages: Annotated[list[AnyMessage], add_messages]

def chat_node(state: State) -> State:
    model = init_chat_model("google_genai:gemini-2.5-flash", api_key = GOOGLE_API_KEY)
    state['messages'] = model.invoke(state['messages'])
    return state

grap_builder = StateGraph(State)
grap_builder.add_node("chat_node", chat_node)
grap_builder.add_edge(START, "chat_node")
grap_builder.add_edge("chat_node", END)
graph = grap_builder.compile(checkpointer = MemorySaver())