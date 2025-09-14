"""
This module initializes and exposes available agent tools for use within the application.
The `get_tools` function returns a list of all available tool functions.
"""

from langchain_core.tools import BaseTool

from .elasticsearch import semantic_offer_search
from .maps import (
    find_nearby_supermarkets,
    geocode_address,
    get_accurate_supermarket_distances,
    get_supermarket_details,
)


async def get_tools() -> list[BaseTool]:
    return [
        semantic_offer_search,
        find_nearby_supermarkets,
        geocode_address,
        get_accurate_supermarket_distances,
        get_supermarket_details,
    ]
