# Architecture

## Technical Stack
- Language: Go 1.24+
- LLM Integrations: LM Studio (OpenAI-compatible), Ollama
- Paradigm: Concurrency-based Execution Loop

## Core Architecture

The Agentic Coder is split into three main components:

1. **Orchestrator (`pkg/orchestrator/agent.go`)**:
   The heart of the application. The Agent tracks conversation history, interacts with the LLM via `getLLMResponseStream()`, and dynamically streams out the AI's step-by-step thoughts. It manages an intelligent execution loop that parses JSON output and invokes the requested tools asynchronously mapped in the registry. It enforces performance tuning with variables like `temperature`, `num_predict`, `top_k`, `top_p`, `repeat_penalty` and `seed`.

2. **LLM Client (`pkg/llm/client.go`)**:
   Provides an abstraction layer to communicate with LLM providers (LM Studio/Ollama). Key implementation details:
   - Uses raw JSON parsing for streaming responses to handle tool call chunks properly
   - Implements tool call accumulation by index (since name and arguments arrive in separate chunks)
   - Custom `UnmarshalJSON` for `ToolCallFunction` to handle both string and map arguments
   - Proper message format with `tool_call_id` for tool result responses

3. **Tool Registry (`pkg/tools/file_tools.go`)**:
   A mapping interface containing the actual implementations of file manipulation, searching (`grep`), web fetching, and shell execution. These tools provide the autonomous agent with complete read/write access to the local user environment.

## Supported Providers
- **LM Studio**: Connect to `http://localhost:1234` (or custom port), use provider `"lmstudio"`
- **Ollama**: Connect to `http://localhost:11434`, use provider `"openai"`

## Execution Flow

```
User Input → Agent.Execute()
    │
    ├─→ getLLMResponseStream() ──→ LLM Client.StreamChat()
    │                                           │
    │    ┌────────────────────────────────────┘
    │    ▼
    │   LLM returns tool call (streaming)
    │    │
    │    ├─→ Tool Call Parsing
    │    │   (accumulate chunks by index)
    │    │   
    │    ├─→ Execute Tool
    │    │   │
    │    │   └─→ Add tool result to conversation
    │    │
    │    └─→ Loop: Send to LLM again
    │
    └─→ Return final response
```

## Tool Calling Implementation

### Streaming Tool Call Handling
The most complex part of the implementation is handling streaming tool calls:

1. **Chunk 1**: Contains `index`, `id`, `name`, and empty `arguments`
2. **Chunk 2**: Contains `index` and `arguments` (as JSON string)

The solution uses index-based accumulation:
- Use `index` (present in all chunks) as the merge key
- Merge name from first chunk with arguments from subsequent chunks
- Parse accumulated arguments as JSON to get parameter map

### Message Format
When sending tool results back to the LLM, the message must include:
- `role: "tool"`
- `tool_call_id`: The ID from the original tool call
- `content`: The tool result

This is required by the OpenAI API specification.
