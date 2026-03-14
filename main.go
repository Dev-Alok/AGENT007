package main

import (
	"context"
	"log"
	"os"

	"agentic-coder/pkg/config"
	"agentic-coder/pkg/orchestrator"
	"agentic-coder/pkg/tools"
	"agentic-coder/tui"
)

// IMPORTANT: All input and output must go through TUI!
// - NO direct fmt.Print, fmt.Println, fmt.Printf
// - NO direct os.Stdout, os.Stderr usage
// - Use TUI for all user-facing output
// - Use log package for internal logging only
// - Use TUI.SetOutput() callback for streaming responses

func main() {
	// Initialize logging to file only (NOT to console)
	logFile, err := os.OpenFile("agent.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err == nil {
		log.SetOutput(logFile)
		log.Println("--- Agent Session Started ---")
		defer logFile.Close()
	}

	// Load configuration
	cfg, err := config.LoadConfig("config.json")
	if err != nil {
		// Cannot use TUI yet - exit with error
		log.Fatalf("Error loading config: %v", err)
		os.Exit(1)
	}

	// Register all tools
	registry := tools.NewRegistry()
	registry.Register("read_file", tools.ReadFileTool())
	registry.Register("write_file", tools.WriteFileTool())
	registry.Register("run_command", tools.RunCommandTool())
	registry.Register("list_directory", tools.ListDirectoryTool())
	registry.Register("search_file", tools.SearchFileTool())
	registry.Register("search_directory", tools.SearchDirectoryTool())
	registry.Register("grep_search", tools.GrepSearchTool())
	registry.Register("replace_file_content", tools.ReplaceFileContentTool())
	registry.Register("read_url", tools.ReadURLTool())
	registry.Register("file_exists", tools.FileExistsTool())
	registry.Register("get_working_directory", tools.GetWorkingDirTool())
	registry.Register("get_env_vars", tools.GetEnvVarsTool())
	registry.Register("glob", tools.GlobTool())
	registry.Register("line_count", tools.LineCountTool())
	registry.Register("file_extension", tools.FileExtensionTool())

	// Create agent
	agent := orchestrator.NewAgentWithTimeout(registry, cfg, 0)

	// Check LLM connection
	err = agent.GetLLMClient().HealthCheck(context.Background())
	connected := err == nil

	// Run TUI - this handles ALL input/output
	tuiApp := tui.NewTUI(agent, cfg, connected)
	if err := tuiApp.Run(); err != nil {
		log.Printf("TUI ended: %v", err)
	}
}
