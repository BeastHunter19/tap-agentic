"""Custom routing logic for the agentic graph.

This module defines the routing functions used to determine the next node.
"""

from typing import Literal

from langgraph.graph import END

from agent.state import OverallState


def custom_tools_condition(state: OverallState) -> Literal["tools", END]:
    """Route to workflow, tools, or END based on tool calls in the state."""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    tool_calls = None
    if last_message:
        # tool_calls may be under 'tool_calls' or 'additional_kwargs' depending on your message format
        if isinstance(last_message, dict):
            tool_calls = last_message.get("tool_calls")
            # Some LLMs put tool calls in 'additional_kwargs' as 'tool_calls' or 'function_call'
            if tool_calls is None and "additional_kwargs" in last_message:
                tool_calls = last_message["additional_kwargs"].get("tool_calls")
        else:
            # fallback for object-like messages
            tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        action_names = [
            a.get("name") for a in state.get("copilotkit", {}).get("actions", [])
        ]
        for call in tool_calls:
            if call.get("name") in action_names:
                return END
        return "tools"
    return END
