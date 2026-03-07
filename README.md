# Agentic Coder

**Agentic Coder** is an experimental AI Agent built in Go for learning purposes. It is designed to demonstrate how Large Language Models (LLMs) can autonomously write code, execute local terminal commands, and navigate the filesystem using tool implementations.

## Overview
This project binds an interactive Terminal CLI to an AI model (like Ollama or OpenAI/LM Studio), giving it access to a registry of tools to help you develop, debug, and understand your local file systems. 

For detailed design, implementation details, and architecture mapping, please see the [Wiki Documentation](./wiki/Architecture.md).

## Usage
Ensure you have a local LLM running (e.g. LM Studio on port `1234`), then run the agent interface:
```bash
go run main.go
```
