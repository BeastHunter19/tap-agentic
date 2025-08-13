FROM node:24-slim

RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g @cablate/mcp-google-map

WORKDIR /app

COPY start-server.sh ./

ENV MCP_SERVER_PORT=3000

EXPOSE ${MCP_SERVER_PORT}

# Not using -f because there is no /health endpoint, so I will accept a 400 response
# from the /mcp endpoint as a valid health check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=5 \
    CMD curl http://localhost:${MCP_SERVER_PORT}/mcp || exit 1

ENTRYPOINT [ "./start-server.sh" ]