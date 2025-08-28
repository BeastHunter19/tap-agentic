import os
from typing import Any, Dict, List, Optional

from elasticsearch import AsyncElasticsearch
from langchain_core.tools import tool

"""Tools for interacting with Elasticsearch to search for offers."""

es_endpoint = os.getenv("ELASTICSEARCH_ENDPOINT")

if not es_endpoint:
    raise RuntimeError("ELASTICSEARCH_ENDPOINT must be set")

es_client = AsyncElasticsearch(hosts=[es_endpoint])


@tool
async def search_offers(
    query: Optional[Dict[str, Any]] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    sort: Optional[List[Dict[str, Any]]] = None,
    size: int = 10,
) -> List[Dict[str, Any]]:
    """Search for current offers in the Elasticsearch index.
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

    Args:
        query: The Elasticsearch query DSL dict (e.g., {"match": {"product": "milk"}}).
        filters: List of filter dicts to apply (e.g., [{"term": {"category": "dairy"}}]).
        sort: List of sort dicts (e.g., [{"price": "asc"}]).
        size: Number of results to return.

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

    # Only return non-expired offers from open supermarkets (example filter)
    # You can adjust these fields as needed for your schema
    # additional_filters = [
    #     {"range": {"valid_until": {"gte": "now"}}},
    #     {"term": {"supermarket_open": True}},
    # ]
    # if "filter" in body["query"]["bool"]:
    #     body["query"]["bool"]["filter"].extend(additional_filters)
    # else:
    #     body["query"]["bool"]["filter"] = additional_filters

    resp = await es_client.search(index="offers", body=body, size=size)
    return [hit["_source"] for hit in resp["hits"]["hits"]]
