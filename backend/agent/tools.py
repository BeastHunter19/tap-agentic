"""Helpers to discover and return MCP tools used by the agent.

This module encapsulates the creation of the `MultiServerMCPClient` and
the asynchronous retrieval of tool descriptors from configured MCP
endpoints. It returns a sequence of tool descriptors that can be bound to
the chat model.
"""

import os

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_tools() -> list[BaseTool]:
    """Discover MCP tools from the configured MCP servers.

    Environment variables used:
    - `ELASTICSEARCH_MCP_ENDPOINT`: URL of the Elasticsearch MCP server.
    - `MAPS_MCP_ENDPOINT`: URL of the Maps MCP server.

    Returns:
        A sequence of tool descriptor objects as returned by the MCP client.

    Raises:
        RuntimeError: If required environment variables are not set.
    """
    es_endpoint = os.getenv("ELASTICSEARCH_MCP_ENDPOINT")
    maps_endpoint = os.getenv("MAPS_MCP_ENDPOINT")

    if not es_endpoint or not maps_endpoint:
        raise RuntimeError(
            "ELASTICSEARCH_MCP_ENDPOINT and MAPS_MCP_ENDPOINT must be set"
        )

    client = MultiServerMCPClient(
        {
            "elasticsearch": {"url": es_endpoint, "transport": "streamable_http"},
            "maps": {"url": maps_endpoint, "transport": "streamable_http"},
        }
    )

    return await client.get_tools()
