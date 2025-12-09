FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04
# Use the official devcontainer base image for Ubuntu 24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Free Pascal, fpc-source and Lazarus non-interactively and clean up
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    fpc \
    fpc-source \
    lazarus \
    php \
    build-essential \
 || true \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Ensure workspace folder exists (devcontainers typically mount the repo)
WORKDIR /workspaces/smolagents

# Use the default non-root user provided by the base image
USER vscode
