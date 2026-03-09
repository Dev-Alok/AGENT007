# Agentic Coder

**Agentic Coder** is an experimental AI Agent built in Go for learning purposes. It is designed to demonstrate how Large Language Models (LLMs) can autonomously write code, execute local terminal commands, and navigate the filesystem using tool implementations.

## Overview
This project binds an interactive Terminal CLI to an AI model (like Ollama), giving it access to a registry of tools to help you develop, debug, and understand your local file systems. 

For detailed design, implementation details, and architecture mapping, please see the [Wiki Documentation](./wiki/Architecture.md).

## Configuration
Before running the agent, you must configure your local LLM settings.

1. Copy the example configuration file:
   ```bash
   cp example-config.json config.json
   ```
2. Open `config.json` and update your `"llm_endpoint"`, `"model"`, and reasoning parameters (`temperature`, `num_predict`, `top_k`, `top_p`, `repeat_penalty`, `seed`). 
*(Note: `config.json` is ignored by git to protect your data.)*

## Usage
Ensure you have a local LLM running (e.g. Ollama `http://localhost:11434`), then open a terminal and run the agent interface:
```bash
go run main.go
```
