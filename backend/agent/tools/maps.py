import os

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

"""Tools related to maps and geolocation."""


async def get_maps_tools() -> list[BaseTool]:
    """Discover MCP tools from the configured MCP server.

    Environment variables used:
    - `MAPS_MCP_ENDPOINT`: URL of the Maps MCP server.

    Returns:
        A sequence of tool descriptor objects as returned by the MCP client.

    Raises:
        RuntimeError: If required environment variable is not set.
    """
    maps_endpoint = os.getenv("MAPS_MCP_ENDPOINT")

    if not maps_endpoint:
        raise RuntimeError("MAPS_MCP_ENDPOINT must be set")

    client = MultiServerMCPClient(
        {
            # "elasticsearch": {"url": es_endpoint, "transport": "streamable_http"},
            "maps": {"url": maps_endpoint, "transport": "streamable_http"},
        }
    )
    return await client.get_tools()
