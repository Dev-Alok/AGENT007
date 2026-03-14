# Agentic Coder - Next Iteration Improvements

## Completed
- Streaming support for LLM responses
- Tool calling with automatic function invocation
- LM Studio support (OpenAI-compatible API)
- Streaming tool call parsing with proper argument accumulation

## In Progress
- Token counting for precise context window management

## Backlog

### High Priority
- Add support for Docker-based Sandboxing to safely execute untrusted `run_command` operations without manual blacklisting
- Create an API switch toggle in `config.json` to seamlessly swap between Anthropic Claude and local Ollama without recompiling

### Medium Priority
- Separate `agent.log` logging into structured JSON traces (e.g. `slog`) for easier ingestion by external debugging tools like Grafana/ELK
- Implement rate-limiting logic (tokens per minute) to natively respect public API throttles if external providers are used

### Low Priority
- Add parallel tool dependency resolution (a directed acyclic graph) so the AI can batch truly independent functions while waiting for sequential blockers
- Build an optional Web UI dashboard out of the new `agent.log` JSON structs to visually trace the AI's Chain-of-Thought paths
