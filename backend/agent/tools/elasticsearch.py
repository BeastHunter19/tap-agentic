"""Semantic search tools for querying offers in Elasticsearch using dense vector similarity.

This module provides:
- A semantic search tool (`semantic_offer_search`) that leverages LangChain's SelfQueryRetriever and ElasticsearchStore.
- Support for natural language queries and filters over offer metadata fields.
- Integration with a local OpenAI-compatible embedding server for query embedding.

The elasticsearch index should be pre-populated with offer data including dense vector embeddings.
"""

import os
from typing import Any, Dict, List, Optional

# import httpx
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings.infinity import InfinityEmbeddings
from langchain_community.query_constructors.elasticsearch import ElasticsearchTranslator
from langchain_core.documents import Document

# from httpx_retries import Retry, RetryTransport
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_elasticsearch import DenseVectorStrategy, ElasticsearchStore
from pydantic import BaseModel, Field

from agent.model import create_model

# Lazy initialization
_embedding_model: Optional[Embeddings] = None
_vector_store: Optional[ElasticsearchStore] = None
_retriever: Optional[SelfQueryRetriever] = None


def _get_embedding_model() -> Embeddings:
    """Return (and lazily initialize) an embedding model."""

    global _embedding_model
    if _embedding_model is None:
        api_url = os.getenv("EMBEDDING_API_URL")
        if not api_url:
            raise RuntimeError("EMBEDDING_API_URL must be set for vector search")
        api_model = os.getenv("EMBEDDING_MODEL_NAME")
        if not api_url:
            raise RuntimeError("EMBEDDING_MODEL_NAME must be set for vector search")

        _embedding_model = InfinityEmbeddings(model=api_model, infinity_api_url=api_url)
    return _embedding_model


def _get_vector_store() -> ElasticsearchStore:
    """Lazily initialize an ElasticsearchStore for semantic search."""

    global _vector_store
    if _vector_store is None:
        es_url = os.getenv("ELASTICSEARCH_ENDPOINT")
        if not es_url:
            raise RuntimeError("ELASTICSEARCH_ENDPOINT must be set for vector search")

        _vector_store = ElasticsearchStore(
            es_url=es_url,
            index_name="offers",
            strategy=DenseVectorStrategy(hybrid=False),
            embedding=_get_embedding_model(),
            query_field="name",
            vector_query_field="embeddings",
        )
    return _vector_store


def elastic_doc_builder(hit: Dict) -> Document:
    """Build a Document from an Elasticsearch hit."""

    print(hit)
    source = hit.get("_source", {})
    if not source:
        return Document(page_content="Missing offer", metadata={})
    exclude_fields = {"embeddings", "flyer_checksum", "name"}
    metadata = {k: v for k, v in source.items() if k not in exclude_fields}
    return Document(page_content=source.get("name", "Missing offer"), metadata=metadata)


def _get_retriever() -> SelfQueryRetriever:
    """Lazily initialize a SelfQueryRetriever for semantic search with filters."""

    global _retriever
    if _retriever is None:
        # Define the attributes that can be used to filter the search
        attributes = [
            AttributeInfo(
                name="name",
                description="Name/description of the product or offer.",
                type="string",
            ),
            AttributeInfo(
                name="price",
                description="Price of the product or offer.",
                type="float",
            ),
            AttributeInfo(
                name="quantity",
                description="Quantity per unit (e.g., 1.0 for 1L).",
                type="float",
            ),
            AttributeInfo(
                name="total_quantity",
                description="Total quantity available. (usually the product of quantity and count)",
                type="float",
            ),
            AttributeInfo(
                name="count",
                description="Number of items in the offer.",
                type="float",
            ),
            AttributeInfo(
                name="uom",
                description='Unit of measure (e.g., "kg", "L").',
                type="string",
            ),
            AttributeInfo(
                name="category",
                description='Category of the product (e.g., "dairy", "produce").',
                type="string",
            ),
            AttributeInfo(
                name="type",
                description='Type of offer or product (e.g., "mozzarella cheese", "chicken meat").',
                type="string",
            ),
            AttributeInfo(
                name="notes",
                description='Additional notes or description (e.g. "requires fidelity card").',
                type="string",
            ),
            AttributeInfo(
                name="source",
                description="Source of the offer (e.g., supermarket name).",
                type="string",
            ),
            # AttributeInfo(
            #     name="flyer_checksum",
            #     description="Checksum of the flyer or source document.",
            #     type="string",
            # ),
            AttributeInfo(
                name="validity_from",
                description="Start date of offer validity.",
                type="date",
            ),
            AttributeInfo(
                name="validity_to",
                description="End date of offer validity.",
                type="date",
            ),
        ]

        _retriever = SelfQueryRetriever.from_llm(
            llm=create_model("google_genai:gemini-2.5-flash", temperature=0.3),
            vectorstore=_get_vector_store(),
            structured_query_translator=ElasticsearchTranslator(),
            document_contents="Description of the offer/product",
            metadata_field_info=attributes,
            enable_limit=True,
            verbose=True,
            search_kwargs={
                "k": 20,
                "doc_builder": elastic_doc_builder,
            },
        )
    return _retriever


class SemanticOfferSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="Natural language description of the user requests and preferences.",
    )


@tool(args_schema=SemanticOfferSearchInput)
async def semantic_offer_search(query: str) -> List[Dict[str, Any]]:
    """Perform dense vector (semantic) search over offers. You can specify both
    an actual query and any number of natural language filters at the same time
    (e.g., "I want offers for fruit still running at the moment below a specific price").
    You can also adjust the number of results to return by specifying it in the query, e.g., "Top 10 offers for ...".
    """
    retriever = _get_retriever()
    results = retriever.invoke(query)
    return [{"name": doc.page_content, **doc.metadata} for doc in results]
