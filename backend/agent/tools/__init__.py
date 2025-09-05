from langchain_core.tools import BaseTool

from .elasticsearch import search_offers
from .maps import (
    find_nearby_supermarkets,
    geocode_address,
    get_accurate_supermarket_distances,
    get_supermarket_details,
)

"""
This module initializes and exposes available agent tools for use within the application.
The `get_tools` function returns a list of all available tool functions.
"""


async def get_tools() -> list[BaseTool]:
    return [
        search_offers,
        find_nearby_supermarkets,
        geocode_address,
        get_accurate_supermarket_distances,
        get_supermarket_details,
    ]
