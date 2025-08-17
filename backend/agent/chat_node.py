"""Chat node implementation used by the agent graph.

This module defines a top-level async-callable `ChatNode` class. The node is
constructed at application startup with the chat model and the discovered
or provided tools injected.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from agent.state import OverallState


class ChatNode:
    """Async callable node that delegates messages to the LLM bound with tools.

    The node updates the `OverallState.messages` list with the response from
    the model. The model and tools are injected at construction time which
    makes the node easy to test and avoids import-time side effects.

    Attributes:
        model: A LangChain `BaseChatModel`.
        tools: List of `BaseTool` tool descriptors discovered from MCP servers or any other source.
    """

    def __init__(self, model: BaseChatModel, tools: list[BaseTool]) -> None:
        """Create a ChatNode.

        Args:
            model: Chat model factory / initializer that implements
                `bind_tools(tools) -> BoundModelProtocol`.
            tools: Sequence of tool descriptors returned by the MCP client.
        """
        self.model = model
        self.tools = tools

    async def __call__(self, state: OverallState) -> OverallState:
        """Execute the chat step for the given state.

        This method is awaited by the graph runner when the node executes.
        Uses the model's async invocation.

        Args:
            state: The current `OverallState` containing the conversation
                messages under `state.messages`.

        Returns:
            state: The (possibly mutated) `OverallState` with updated messages.
        """
        state["messages"] = await self.model.bind_tools(self.tools).ainvoke(
            state["messages"]
        )
        return state
