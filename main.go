package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"agentic-coder/pkg/config"
	"agentic-coder/pkg/orchestrator"
	"agentic-coder/pkg/tools"
	"agentic-coder/tui"
)

const (
	colorReset  = "\033[0m"
	colorGreen  = "\033[32m"
	colorYellow = "\033[33m"
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

	if err := agent.GetLLMClient().HealthCheck(context.Background()); err != nil {
		fmt.Printf("%sWarning: Could not connect to %s at %s: %v%s\n", colorYellow, cfg.Provider, cfg.LLMEndpoint, err, colorReset)
		fmt.Println("Attempting to continue anyway...")
	} else {
		fmt.Printf("%s✓ Connection to %s validated successfully.%s\n", colorGreen, cfg.Provider, colorReset)
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
