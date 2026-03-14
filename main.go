package main

import (
	"fmt"
	"log"
	"os"

	"agentic-coder/pkg/config"
	"agentic-coder/pkg/orchestrator"
	"agentic-coder/pkg/tools"
	"agentic-coder/tui"
)

func main() {
	logFile, err := os.OpenFile("agent.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err == nil {
		log.SetOutput(logFile)
		log.Println("--- Agent Session Started ---")
		defer logFile.Close()
	} else {
		fmt.Printf("Warning: Failed to open agent.log: %v\n", err)
	}

	cfg, err := config.LoadConfig("config.json")
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		os.Exit(1)
	}

	registry := tools.NewRegistry()
	registry.Register("read_file", tools.ReadFileTool())
	registry.Register("write_file", tools.WriteFileTool())
	registry.Register("run_command", tools.RunCommandTool())
	registry.Register("list_directory", tools.ListDirectoryTool())
	registry.Register("search_file", tools.SearchFileTool())
	registry.Register("grep_search", tools.GrepSearchTool())
	registry.Register("replace_file_content", tools.ReplaceFileContentTool())
	registry.Register("read_url", tools.ReadURLTool())

	agent := orchestrator.NewAgentWithTimeout(registry, cfg, 0)

	if err := agent.LoadHistory(".agent_history.json"); err != nil {
		fmt.Printf("Note: Could not load previous history: %v\n", err)
	}

	if err := agent.GetLLMClient().HealthCheck(nil); err != nil {
		fmt.Printf("Warning: Could not connect to %s at %s: %v\n", cfg.Provider, cfg.LLMEndpoint, err)
		fmt.Println("Attempting to continue anyway...")
	} else {
		fmt.Printf("✓ Connection to %s validated successfully.\n", cfg.Provider)
	}

	t := tui.NewTUI(agent, cfg)
	if err := t.Run(); err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	if err := agent.SaveHistory(".agent_history.json"); err != nil {
		fmt.Printf("Error saving history: %v\n", err)
	}
}
