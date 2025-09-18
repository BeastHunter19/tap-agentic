FROM mcr.microsoft.com/devcontainers/typescript-node:24

# Set workdir to the frontend app folder inside the workspace mount
WORKDIR /workspaces/tap-agentic/frontend

# Ensure .next directory exists and is owned by the node user
RUN mkdir .next && chown node:node .next

# Copy only package files first for better Docker cache usage
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install
