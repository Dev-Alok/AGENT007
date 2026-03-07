# Agentic Coder - Next Iteration Improvements

- Implement proper Token Counting using a lightweight BPE Tokenizer for precise Context Window management.
- Add support for Docker-based Sandboxing to safely execute untrusted `run_command` operations without manual blacklisting.
- Create an API switch toggle in `config.json` to seamlessly swap between Anthropic Claude and local LM Studio without recompiling.
- Separate `agent.log` logging into structured JSON traces (e.g. `slog`) for easier ingestion by external debugging tools like Grafana/ELK.
- Implement rate-limiting logic (tokens per minute) to natively respect public API throttles if external providers are used.
- Add parallel tool dependency resolution (a directed acyclic graph) so the AI can batch truly independent functions while waiting for sequential blockers.
- Build an optional Web UI dashboard out of the new `agent.log` JSON structs to visually trace the AI's Chain-of-Thought paths.
