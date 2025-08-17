"""Agent runtime state used by the LangGraph StateGraph.

This module defines the `OverallState` dataclass used as the root state
for the graph.
"""

from typing import Annotated, List

from copilotkit import CopilotKitState
from langgraph.graph.message import AnyMessage, add_messages


class OverallState(CopilotKitState):
    """The top-level state object passed through the graph.

    Attributes:
        messages: A list of messages exchanged in the conversation. The
            `add_messages` annotation is a reducer that appends new
            messages to the list.
    """

    messages: Annotated[List[AnyMessage], add_messages]
