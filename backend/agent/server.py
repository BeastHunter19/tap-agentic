"""FastAPI server and application setup for the agent.

This module creates the FastAPI `app`, wires the CopilotKit endpoint during
startup by building the agent graph, and exposes a simple `/health`
endpoint. The `main` function is provided for local development.
"""

import uvicorn
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from fastapi import FastAPI

from agent.graph import build_graph

app = FastAPI()


@app.on_event("startup")
async def startup_event() -> None:
    """FastAPI startup handler that builds the agent graph and registers the SDK.

    The graph is initialized asynchronously (model creation and tool
    discovery) and then used to construct a `LangGraphAgent` which is
    registered with the CopilotKit FastAPI integration.
    """
    graph = await build_graph()
    sdk = CopilotKitRemoteEndpoint(
        agents=[
            LangGraphAgent(
                name="offers_agent",
                description=(
                    "An agent that gives shopping recommendations for the best supermarkets."
                ),
                graph=graph,
            )
        ],
    )
    add_fastapi_endpoint(app, sdk, "/copilotkit")


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check used by container orchestration.

    Returns a small JSON payload. Keep this lightweight to ensure the
    container health check has minimal overhead.
    """
    return {"status": "ok"}


def main() -> None:
    """Run the Uvicorn development server.

    Note: In production the application is typically run via the Uvicorn
    ASGI server (for example `uvicorn agent.server:app`). This helper is
    convenient for local development.
    """
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
