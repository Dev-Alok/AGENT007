# Agentic Coder

**Agentic Coder** is an experimental AI Agent built in Go for learning purposes. It is designed to demonstrate how Large Language Models (LLMs) can autonomously write code, execute local terminal commands, and navigate the filesystem using tool implementations.

## Overview
This project binds an interactive Terminal CLI to an AI model (like Ollama or LM Studio), giving it access to a registry of tools to help you develop, debug, and understand your local file systems. 

For detailed design, implementation details, and architecture mapping, please see the [Wiki Documentation](./wiki/Architecture.md).

## Features
- **Streaming Support**: Real-time streaming of LLM responses
- **Tool Calling**: Automatic function calling with tool result handling
- **Multi-Provider Support**: Works with Ollama and LM Studio (OpenAI-compatible API)
- **Concurrent Execution**: Parallel tool execution when multiple tools are called

## Supported Tools
- `list_directory` - List files and directories
- `read_file` - Read file contents
- `write_file` - Write content to files
- `replace_file_content` - Search and replace file content
- `grep_search` - Search for patterns in files
- `search_file` - Find files by name
- `run_command` - Execute shell commands
- `read_url` - Fetch content from web URLs

## Configuration
Before running the agent, you must configure your local LLM settings.

1. Copy the example configuration file:
   ```bash
   cp example-config.json config.json
   ```
2. Open `config.json` and update your settings:
   - `"llm_endpoint"`: Your LLM API endpoint (e.g., `http://localhost:1234` for LM Studio, `http://localhost:11434` for Ollama)
   - `"model"`: The model name to use
   - `"provider"`: Either `"lmstudio"` or `"openai"` (for Ollama)
   - Other parameters: `temperature`, `num_predict`, `top_k`, `top_p`, `repeat_penalty`, `seed`

*(Note: `config.json` is ignored by git to protect your data.)*

## Usage
Ensure you have a local LLM running (e.g., LM Studio or Ollama), then open a terminal and run the agent interface:
```bash
go run main.go
```

### Commands
- Type your request and press Enter
- `/help` - Show help
- `/clear` - Clear screen
- `/config` - Show configuration
- `/exit` - Exit the application

## Architecture
For detailed architecture information, see [Wiki Documentation](./wiki/Architecture.md).

## Technical Details
The agent implements a Think-Act-Observation loop:
1. **Think**: LLM decides what tool to use
2. **Act**: Execute the tool and get results
3. **Observation**: Send results back to LLM
4. Repeat until task is complete

### Key Implementation Details
- **Streaming Tool Calls**: Uses raw JSON parsing to handle streaming tool call responses where arguments arrive as partial strings
- **Tool Call Accumulation**: Merges tool call chunks using index-based accumulation
- **Message Format**: Properly includes `tool_call_id` when sending tool results back to the LLM (required by OpenAI API)
