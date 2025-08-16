from typing import Annotated
from langgraph.graph.message import add_messages, AnyMessage
from copilotkit import CopilotKitState

class OverallState(CopilotKitState):
    messages: Annotated[list[AnyMessage], add_messages]