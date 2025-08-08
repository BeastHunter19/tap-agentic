from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from agent import graph
from fastapi import FastAPI
import uvicorn

app = FastAPI()

sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name = "offers_agent",
            description = "An agent that gives shopping recommendations for the best supermarkets.",
            graph = graph,
        )
    ],
)

add_fastapi_endpoint(app, sdk, "/copilotkit")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

def main():
    """Run the uvicorn server."""
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()