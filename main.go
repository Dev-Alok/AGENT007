package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"agentic-coder/pkg/config"
	"agentic-coder/pkg/orchestrator"
	"agentic-coder/pkg/tools"
)

var (
	Colors = struct {
		Reset   string
		Bold    string
		Purple  string
		Cyan    string
		Green   string
		Yellow  string
		Red     string
		Grey    string
	}{
		Reset:  "\033[0m",
		Bold:   "\033[1m",
		Purple: "\033[35m",
		Cyan:   "\033[36m",
		Green:  "\033[32m",
		Yellow: "\033[33m",
		Red:    "\033[31m",
		Grey:   "\033[90m",
	}
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

	if err := agent.GetLLMClient().HealthCheck(context.Background()); err != nil {
		fmt.Printf("Warning: Could not connect to %s at %s: %v\n", cfg.Provider, cfg.LLMEndpoint, err)
	} else {
		fmt.Printf("%s✓ Connected to %s%s\n", Colors.Green, cfg.Provider, Colors.Reset)
	}

	printHeader(cfg)

	scanner := bufio.NewScanner(os.Stdin)
	
	for {
		fmt.Printf("\n%s%s➜%s ", Colors.Bold, Colors.Green, Colors.Reset)
		
		if !scanner.Scan() {
			break
		}
		
		input := scanner.Text()
		input = strings.TrimSpace(input)
		if input == "" {
			continue
		}
		
		switch input {
		case "exit", "quit", "q":
			return
		case "/help", "help":
			printHelp()
		case "/clear", "clear":
			printHeader(cfg)
			continue
		case "/config", "config":
			printConfig(cfg)
			continue
		}

		// Execute task
		ctx := context.Background()
		
		_, err := agent.Execute(ctx, input)
		if err != nil {
			printError(err.Error())
		}
	}
}

func printHeader(cfg *config.AgentConfig) {
	fmt.Print("\033[2J\033[H")
	fmt.Printf("%s╔══════════════════════════════════════════════════════════════════╗%s\n", Colors.Grey, Colors.Reset)
	fmt.Printf("%s║%s  %s🤖 Agentic Coder - LM Studio Edition                     %s║%s\n", Colors.Grey, Colors.Reset, Colors.Bold+Colors.Cyan, Colors.Grey, Colors.Reset)
	fmt.Printf("%s║%s  %sStreaming AI Coding Agent                                 %s║%s\n", Colors.Grey, Colors.Reset, Colors.Grey, Colors.Grey, Colors.Reset)
	fmt.Printf("%s╠══════════════════════════════════════════════════════════════════╣%s\n", Colors.Grey, Colors.Reset)
	fmt.Printf("%s║%s  Provider: %s%-12s  Model:     %s%-20s %s║%s\n", 
		Colors.Grey, Colors.Reset, Colors.Cyan, cfg.Provider, Colors.Cyan, cfg.Model, Colors.Grey, Colors.Reset)
	fmt.Printf("%s║%s  Endpoint: %s%-45s %s║%s\n", 
		Colors.Grey, Colors.Reset, Colors.Cyan, cfg.LLMEndpoint, Colors.Grey, Colors.Reset)
	fmt.Printf("%s╚══════════════════════════════════════════════════════════════════╝%s\n", Colors.Grey, Colors.Reset)
	fmt.Println()
	fmt.Println(Colors.Grey + "Commands:" + Colors.Reset + " /help | /clear | /config | /exit")
	fmt.Println()
}

func printBlock(title, content string, titleColor string) {
	width := 68
	lines := wrapText(content, width-2)

	fmt.Printf("%s┌", Colors.Grey)
	for i := 0; i < width; i++ {
		fmt.Print("─")
	}
	fmt.Printf("┐%s\n", Colors.Reset)

	fmt.Printf("%s│%s %s%s%s\n", Colors.Grey, Colors.Reset, titleColor, title, Colors.Reset)
	fmt.Printf("%s│", Colors.Grey)
	for i := 0; i < width; i++ {
		fmt.Print("─")
	}
	fmt.Printf("│%s\n", Colors.Reset)

	for _, line := range lines {
		padding := width - len(stripANSI(line)) - 2
		fmt.Printf("%s│ %s%s%s%s%s│%s\n", Colors.Grey, Colors.Reset, line, strings.Repeat(" ", padding), Colors.Grey, Colors.Reset)
	}

	fmt.Printf("%s└", Colors.Grey)
	for i := 0; i < width; i++ {
		fmt.Print("─")
	}
	fmt.Printf("┘%s", Colors.Reset)
}

func printError(content string) {
	width := 68
	lines := wrapText(content, width-2)

	fmt.Printf("%s┌", Colors.Grey)
	for i := 0; i < width; i++ {
		fmt.Print("─")
	}
	fmt.Printf("┐%s\n", Colors.Reset)

	fmt.Printf("%s│%s %s✗ Error%s\n", Colors.Grey, Colors.Reset, Colors.Red, Colors.Reset)
	fmt.Printf("%s│", Colors.Grey)
	for i := 0; i < width; i++ {
		fmt.Print("─")
	}
	fmt.Printf("│%s\n", Colors.Reset)

	for _, line := range lines {
		padding := width - len(stripANSI(line)) - 2
		fmt.Printf("%s│ %s%s%s%s%s│%s\n", Colors.Grey, Colors.Reset, line, strings.Repeat(" ", padding), Colors.Grey, Colors.Reset)
	}

	fmt.Printf("%s└", Colors.Grey)
	for i := 0; i < width; i++ {
		fmt.Print("─")
	}
	fmt.Printf("┘%s", Colors.Reset)
}

func printHelp() {
	fmt.Printf("\n%s%s┌─ Help ────────────────────────────────────────────────────────┐%s\n", Colors.Bold, Colors.Grey, Colors.Reset)
	fmt.Printf("%s%s│                                                               │%s\n", Colors.Grey, Colors.Reset, Colors.Reset)
	fmt.Printf("%s%s│  Commands:                                                     │%s\n", Colors.Grey, Colors.Reset, Colors.Reset)
	fmt.Printf("%s%s│    /help      - Show this help                                │%s\n", Colors.Grey, Colors.Reset, Colors.Reset)
	fmt.Printf("%s%s│    /clear     - Clear screen                                  │%s\n", Colors.Grey, Colors.Reset, Colors.Reset)
	fmt.Printf("%s%s│    /config   - Show configuration                            │%s\n", Colors.Grey, Colors.Reset, Colors.Reset)
	fmt.Printf("%s%s│    /exit     - Exit the application                           │%s\n", Colors.Grey, Colors.Reset, Colors.Reset)
	fmt.Printf("%s%s│                                                               │%s\n", Colors.Grey, Colors.Reset, Colors.Reset)
	fmt.Printf("%s%s│  Type your request and press Enter                          │%s\n", Colors.Grey, Colors.Reset, Colors.Reset)
	fmt.Printf("%s%s│  The agent will use tools automatically                      │%s\n", Colors.Grey, Colors.Reset, Colors.Reset)
	fmt.Printf("%s%s└───────────────────────────────────────────────────────────────┘%s\n", Colors.Grey, Colors.Reset, Colors.Reset)
}

func printConfig(cfg *config.AgentConfig) {
	fmt.Printf("\n%s%s┌─ Configuration ────────────────────────────────────────────────┐%s\n", Colors.Bold, Colors.Grey, Colors.Reset)
	fmt.Printf("%s%s│                                                               │%s\n", Colors.Grey, Colors.Reset, Colors.Reset)
	fmt.Printf("%s%s│  Provider:    %s%-40s %s│%s\n", Colors.Grey, Colors.Reset, Colors.Cyan, cfg.Provider, Colors.Grey, Colors.Reset)
	fmt.Printf("%s%s│  Model:       %s%-40s %s│%s\n", Colors.Grey, Colors.Reset, Colors.Cyan, cfg.Model, Colors.Grey, Colors.Reset)
	fmt.Printf("%s%s│  Endpoint:    %s%-40s %s│%s\n", Colors.Grey, Colors.Reset, Colors.Cyan, cfg.LLMEndpoint, Colors.Grey, Colors.Reset)
	fmt.Printf("%s%s│  Context:     %s%-40d %s│%s\n", Colors.Grey, Colors.Reset, Colors.Cyan, cfg.ContextWindow, Colors.Grey, Colors.Reset)
	fmt.Printf("%s%s│  Temperature: %s%-40.2f %s│%s\n", Colors.Grey, Colors.Reset, Colors.Cyan, cfg.Temperature, Colors.Grey, Colors.Reset)
	fmt.Printf("%s%s│  Timeout:     %s%-40d %s│%s\n", Colors.Grey, Colors.Reset, Colors.Cyan, cfg.LLMTimeoutSeconds, Colors.Grey, Colors.Reset)
	fmt.Printf("%s%s└───────────────────────────────────────────────────────────────┘%s\n", Colors.Grey, Colors.Reset, Colors.Reset)
}

func wrapText(text string, width int) []string {
	clean := stripANSI(text)
	
	if len(clean) <= width {
		return []string{clean}
	}
	
	var lines []string
	words := strings.Fields(clean)
	currentLine := ""
	
	for _, word := range words {
		if len(currentLine)+len(word)+1 > width {
			if currentLine != "" {
				lines = append(lines, currentLine)
			}
			currentLine = word
		} else {
			if currentLine != "" {
				currentLine += " "
			}
			currentLine += word
		}
	}
	if currentLine != "" {
		lines = append(lines, currentLine)
	}
	
	return lines
}

func stripANSI(s string) string {
	result := strings.Builder{}
	for i := 0; i < len(s); i++ {
		if s[i] == '\033' {
			for i < len(s) && s[i] != 'm' {
				i++
			}
			continue
		}
		result.WriteByte(s[i])
	}
	return result.String()
}
