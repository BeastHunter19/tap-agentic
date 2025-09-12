import os
from typing import Any, Dict, List, Optional

from elasticsearch import AsyncElasticsearch
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.model import create_model
from agent.prompts import generate_dsl_instructions_template

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


# ---- Pydantic schemas for custom tools ----


class GenerateDSLInput(BaseModel):
    query: str = Field(..., description="User's natural language search request.")
    size: int = Field(10000, description="Number of results to return.")


# --- Custom tools for Elasticsearch ---

# Lazy LLM initialization
_dsl_llm_json: Optional[Any] = None


def _get_dsl_llm_json():
    global _dsl_llm_json
    if _dsl_llm_json is None:
        _dsl_llm = create_model("google_genai:gemini-2.5-flash", temperature=0.3)
        # Minimal JSON schema for forcing JSON mode
        _dsl_schema = {
            "title": "ElasticsearchDSL",
            "type": "object",
            "properties": {
                "dsl": {
                    "type": "object",
                    "description": "Elasticsearch DSL query to be used in the 'query' field of an Elasticsearch search.",
                }
            },
            "required": ["dsl"],
        }
        _dsl_llm_json = _dsl_llm.with_structured_output(_dsl_schema)
    return _dsl_llm_json


@tool(args_schema=GenerateDSLInput)
async def search_offers(query: str, size: int = 10000) -> List[Dict[str, Any]]:
    """
    Search for offers in Elasticsearch using a natural language query.
    This tool combines DSL generation and execution to return matching offers.
    Be sure to only include the essential parts of the request that relate to
    the products or categories to search for. You can specify the number of results
    to return with the `size` parameter, but ONLY if the user explicitly requests it
    or when you want to show the results directly to the user; otherwise, if the results
    are to be processed internally (e.g., for filtering or ranking), keep the default size.
    """
    model = _get_dsl_llm_json()
    chain = generate_dsl_instructions_template | model
    result = await chain.ainvoke({"query": query})

    client = _get_es_client()
    body = {"query": result["dsl"]}
    resp = await client.search(index="offers", body=body, size=size)
    return [hit["_source"] for hit in resp["hits"]["hits"]]


