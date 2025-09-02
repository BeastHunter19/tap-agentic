import os
from typing import Any, Dict, List, Optional

from elasticsearch import AsyncElasticsearch
from langchain_core.tools import tool
from pydantic import BaseModel, Field

"""Tools for interacting with Elasticsearch to search for offers."""

# Lazy client creation to avoid initializing when not used
_es_client: Optional[AsyncElasticsearch] = None


def _get_es_client() -> AsyncElasticsearch:
    endpoint = os.getenv("ELASTICSEARCH_ENDPOINT")
    if not endpoint:
        raise RuntimeError("ELASTICSEARCH_ENDPOINT must be set")
    global _es_client
    if _es_client is None:
        _es_client = AsyncElasticsearch(hosts=[endpoint])
    return _es_client


# ---- Pydantic schema for search_offers ----


class SearchOffersInput(BaseModel):
    """Inputs for searching the offers index."""

    query: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Elasticsearch query DSL (e.g., {'match': {'product': 'milk'}}).",
    )
    filters: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of filter clauses (e.g., [{'term': {'category': 'dairy'}}]).",
    )
    sort: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Sort directives (e.g., [{'price': 'asc'}]).",
    )
    size: int = Field(10, description="Maximum number of results to return.")


@tool(args_schema=SearchOffersInput)
async def search_offers(
    query: Optional[Dict[str, Any]] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    sort: Optional[List[Dict[str, Any]]] = None,
    size: int = 10,
) -> List[Dict[str, Any]]:
    """Search for offers in the Elasticsearch index.
    The available fields are:
    - name (text): Name of the product or offer.
    - price (float): Price of the product or offer.
    - quantity (float): Quantity per unit (e.g., 1.0 for 1L).
    - total_quantity (float): Total quantity available.
    - count (float): Number of items in the offer.
    - uom (keyword): Unit of measure (e.g., "kg", "L", "each").
    - category (keyword): Category of the product (e.g., "dairy", "produce").
    - type (keyword): Type of offer or product.
    - notes (text): Additional notes or description.
    - source (keyword): Source of the offer (e.g., supermarket name).
    - flyer_checksum (keyword): Checksum of the flyer or source document.
    - validity_from (date): Start date of offer validity.
    - validity_to (date): End date of offer validity.

    Returns:
        A list of offers matching the search criteria, each including
        details such as product name, price, validity period, and
        supermarket information.
    """
    # Build the query body
    body: Dict[str, Any] = {"query": {"bool": {}}}
    if query:
        body["query"]["bool"]["must"] = query
    if filters:
        body["query"]["bool"]["filter"] = filters
    if sort:
        body["sort"] = sort

    client = _get_es_client()
    resp = await client.search(index="offers", body=body, size=size)
    return [hit["_source"] for hit in resp["hits"]["hits"]]
