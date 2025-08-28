"""
Stub tool for workflow routing.

This module defines a tool that does not perform any action, but serves as an explicit signal
for the LLM to indicate when the user wants to trigger the full shopping plan workflow.
By calling this tool, the LLM enables the agent graph to route execution from a flexible,
agentic chat/tool phase to a deterministic, multi-step workflow. This pattern avoids extra
LLM calls for intent detection and makes routing robust and transparent.
"""

from langchain_core.tools import tool


@tool
async def start_shopping_workflow() -> None:
    """Avvia il flusso di lavoro completo per pianificare la spesa.
    Usa questo strumento quando l'utente desidera ricevere un piano dettagliato
    su dove acquistare tutti gli articoli della sua lista, considerando offerte
    e supermercati raggiungibili.
    """
    pass
