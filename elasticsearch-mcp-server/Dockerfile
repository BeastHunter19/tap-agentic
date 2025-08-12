FROM docker.elastic.co/mcp/elasticsearch:latest

RUN apk update && apk add curl

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=5 \
    CMD curl --fail http://localhost:8080/ping || exit 1

EXPOSE 8080

ENTRYPOINT [ "/usr/local/bin/elasticsearch-core-mcp-server" ]