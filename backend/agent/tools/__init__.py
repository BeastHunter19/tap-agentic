from langchain_core.tools import BaseTool

from .elasticsearch import search_offers
from .maps import get_maps_tools
from .workflow import start_shopping_workflow

"""
This module initializes and exposes available agent tools for use within the application.
The `get_tools` function returns a list of all available tool functions.
"""


async def get_tools() -> list[BaseTool]:
    maps_tools = await get_maps_tools()
    return [search_offers, start_shopping_workflow, *maps_tools]
