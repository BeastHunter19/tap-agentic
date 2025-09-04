"""Chat node implementation used by the agent graph.

This module defines a top-level async-callable `ChatNode` class. The node is
constructed at application startup with the chat model and the discovered
or provided tools injected.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import trim_messages
from langchain_core.tools import BaseTool

from agent.prompts import assistant_instructions_template, get_current_date_time
from agent.state import OverallState


class ChatNode:
    """Async callable node that delegates messages to the LLM bound with tools.

    The node updates the `OverallState.messages` list with the response from
    the model. The model and tools are injected at construction time which
    makes the node easy to test and avoids import-time side effects.

    Attributes:
        model_with_tools: The chat model bound with the tools.
        trimmer: Message trimmer to avoid having infinite context growth.
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[BaseTool],
        max_history_tokens: int = 1048576,
    ) -> None:
        """Create a ChatNode.

        Args:
            model: LangChain chat model to use as the assistant.
            tools: Sequence of tool descriptors.
            max_history_tokens: Maximum number of tokens to keep in the
                conversation history. Older messages are dropped.
        """
        self.backend_tools = tools
        self.model_with_tools = model.bind_tools(tools)
        self.trimmer = trim_messages(
            max_tokens=max_history_tokens,
            strategy="last",
            token_counter=model,
            include_system=True,
            allow_partial=False,
            start_on="human",
            end_on=("human", "tool"),
        )

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

        trimmed_messages = await self.trimmer.ainvoke(state["messages"])
        prompt = await assistant_instructions_template.ainvoke(
            {
                "messages": trimmed_messages,
                "current_date_time": get_current_date_time(),
            }
        )

        # Bind frontend actions as tools if available
        frontend_actions = state.get("copilotkit", {}).get("actions", [])
        tools = self.backend_tools + frontend_actions
        self.model_with_tools = self.model_with_tools.bind_tools(tools)

        response = await self.model_with_tools.ainvoke(prompt)
        return {"messages": [response]}
