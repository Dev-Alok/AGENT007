package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"agentic-coder/pkg/llm"
	"agentic-coder/pkg/orchestrator"
	"agentic-coder/pkg/tools"
)

func main() {
	// Command-line flags for configuration (Go's flag package is lightweight, no dependencies)
	llmEndpoint := flag.String("llm-endpoint", "http://localhost:11434", "LLM API endpoint")
	model := flag.String("model", "llama2", "Model name to use")
	contextWindow := flag.Int("context-window", 5000, "Token context window size")
	useOpenAI := flag.Bool("openai", false, "Use OpenAI-compatible format (for LM Studio)")
	flag.Parse()

	// Initialize signal handling for graceful shutdown
	// Using Go channels for signal notification instead of event handlers (unlike .NET/JS)
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Create tool registry and register built-in tools
	// Map-based registration allows O(1) lookup during execution
	registry := tools.NewRegistry()
	registry.Register("read_file", tools.ReadFileTool())
	registry.Register("write_file", tools.WriteFileTool())
	registry.Register("run_command", tools.RunCommandTool())
	registry.Register("list_directory", tools.ListDirectoryTool())

	// Create orchestrator with context for timeout management
	// Context allows propagating cancellation across goroutines (Go idiom)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var agent *orchestrator.Agent
	if *useOpenAI {
		agent = orchestrator.NewAgentWithAPIType(registry, *contextWindow, *llmEndpoint, *model, llm.APIOpenAI)
	} else {
		agent = orchestrator.NewAgent(registry, *contextWindow, *llmEndpoint, *model)
	}

	fmt.Printf("Agentic Coder initialized. Connecting to %s (API: %v). Enter your task:\n",
		*llmEndpoint, map[bool]string{true: "OpenAI/LM Studio", false: "Ollama"}[*useOpenAI])

	fmt.Println("Agentic Coder initialized. Enter your task:")

	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			input := scanner.Text()
			if input == "exit" || input == "quit" {
				cancel()
				return
			}
			if input == "" {
				continue
			}

			// Create new context for each task with timeout
			taskCtx, taskCancel := context.WithTimeout(ctx, 5*time.Minute)

			// Execute task in goroutine to avoid blocking main loop
			go func() {
				result, err := agent.Execute(taskCtx, input)
				if err != nil {
					fmt.Printf("Error: %v\n", err)
				} else {
					fmt.Printf("\nResult:\n%s\n\n", result)
				}
				taskCancel()
			}()

			// Wait for task completion or cancellation
			select {
			case <-taskCtx.Done():
				if taskCtx.Err() == context.DeadlineExceeded {
					fmt.Println("Task timed out")
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	// Wait for shutdown signal or completion
	select {
	case <-sigChan:
		fmt.Println("\nShutting down...")
	case <-ctx.Done():
	}

	cancel()
	wg.Wait()
}
