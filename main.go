package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"agentic-coder/pkg/config"
	"agentic-coder/pkg/llm"
	"agentic-coder/pkg/orchestrator"
	"agentic-coder/pkg/tools"
)

func main() {
	// Set up file logger for internal system tracing
	logFile, err := os.OpenFile("agent.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err == nil {
		log.SetOutput(logFile)
		log.Println("--- Agent Session Started ---")
		defer logFile.Close()
	} else {
		fmt.Printf("Warning: Failed to open agent.log: %v\n", err)
	}

	// Load Application Configuration
	cfg, err := config.LoadConfig("config.json")
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		os.Exit(1)
	}

	// Use variables from config
	llmEndpoint := cfg.LLMEndpoint
	model := cfg.Model
	contextWindow := cfg.ContextWindow
	useOpenAI := cfg.UseOpenAI
	lmStudioPort := cfg.LMStudioPort
	lmStudioAPIKey := cfg.LMStudioAPIKey
	llmTimeout := cfg.LLMTimeoutSeconds

	// Initialize signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Create tool registry and register built-in tools
	registry := tools.NewRegistry()
	registry.Register("read_file", tools.ReadFileTool())
	registry.Register("write_file", tools.WriteFileTool())
	registry.Register("run_command", tools.RunCommandTool())
	registry.Register("list_directory", tools.ListDirectoryTool())
	registry.Register("search_file", tools.SearchFileTool())
	registry.Register("grep_search", tools.GrepSearchTool())
	registry.Register("replace_file_content", tools.ReplaceFileContentTool())
	registry.Register("read_url", tools.ReadURLTool())

	// Create orchestrator with context for timeout management
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Adjust endpoint if LM Studio is used and endpoint is default
	if useOpenAI && llmEndpoint == "http://localhost:11434" {
		llmEndpoint = fmt.Sprintf("http://localhost:%d", lmStudioPort)
	}

	apiKey := ""
	if useOpenAI && lmStudioAPIKey != "" {
		apiKey = lmStudioAPIKey
	}

	var agent *orchestrator.Agent
	if useOpenAI {
		agent = orchestrator.NewAgentWithTimeoutAndAPIType(registry, contextWindow, llmEndpoint, apiKey, model, llm.APIOpenAI, time.Duration(llmTimeout)*time.Second)
	} else {
		agent = orchestrator.NewAgentWithTimeout(registry, contextWindow, llmEndpoint, apiKey, model, time.Duration(llmTimeout)*time.Second)
	}

	// Try to load state context
	if err := agent.LoadHistory(".agent_history.json"); err != nil {
		fmt.Printf("Note: Could not load previous history: %v\n", err)
	}

	// Validate connection
	if useOpenAI {
		if err := agent.GetLLMClient().HealthCheck(context.Background()); err != nil {
			fmt.Printf("Warning: Could not connect to LM Studio: %v\n", err)
			fmt.Println("Attempting to continue anyway...")
		} else {
			fmt.Println("Connection validated successfully.")
		}
	}

	fmt.Println("\n=======================================================")
	fmt.Println("🤖 Agentic Coder initialized.")
	fmt.Println("Type 'exit' or 'quit' to close.")
	fmt.Println("=======================================================")

	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		scanner := bufio.NewScanner(os.Stdin)
		fmt.Print("\n\033[32m👤 You: \033[0m")
		for scanner.Scan() {
			input := scanner.Text()
			if input == "exit" || input == "quit" {
				cancel()
				return
			}
			if strings.TrimSpace(input) == "" {
				fmt.Print("\n\033[32m👤 You: \033[0m")
				continue
			}

			fmt.Println("\n\033[36m🤖 Thinking...\033[0m")

			// Create new context for each task with timeout
			taskCtx, taskCancel := context.WithTimeout(ctx, 5*time.Minute)

			// Execute task in goroutine to avoid blocking main loop
			go func() {
				_, err := agent.Execute(taskCtx, input)
				if err != nil {
					fmt.Printf("\033[31mError: %v\033[0m\n", err)
				}
				taskCancel()
			}()

			// Wait for task completion or cancellation
			select {
			case <-taskCtx.Done():
				if taskCtx.Err() == context.DeadlineExceeded {
					fmt.Println("\033[31mTask timed out\033[0m")
				}
			case <-ctx.Done():
				return
			}

			// Show prompt again after task is complete
			fmt.Print("\n\033[32m👤 You: \033[0m")
		}
	}()

	// Wait for shutdown signal or completion
	select {
	case <-sigChan:
		fmt.Println("\nShutting down...")
	case <-ctx.Done():
	}

	// Save history before exiting
	fmt.Println("Saving conversation history...")
	if err := agent.SaveHistory(".agent_history.json"); err != nil {
		fmt.Printf("Error saving history: %v\n", err)
	}

	cancel()
	wg.Wait()
}
