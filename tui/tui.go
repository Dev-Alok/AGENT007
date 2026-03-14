package tui

// IMPORTANT: This is the SOLE entry point for all user input and output!
// - All terminal output MUST go through TUI methods (print, displayMarkdown, etc.)
// - NO direct fmt.Print, fmt.Println anywhere in this package
// - All output flows through TUI for consistent formatting
// - Input is read via TUI and passed to agent

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"agentic-coder/pkg/config"
	"agentic-coder/pkg/orchestrator"
	"agentic-coder/tui/renderer"
	"charm.land/lipgloss/v2"
)

var (
	accentStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("86"))
	dimStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	promptStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("154")).Bold(true)
)

type TUI struct {
	agent          *orchestrator.Agent
	cfg            *config.AgentConfig
	done           chan struct{}
	markdownRender *renderer.MarkdownRenderer
	connected      bool
}

func NewTUI(agent *orchestrator.Agent, cfg *config.AgentConfig, connected bool) *TUI {
	mr, _ := renderer.NewAutoStyleRenderer()
	return &TUI{
		agent:          agent,
		cfg:            cfg,
		done:           make(chan struct{}),
		markdownRender: mr,
		connected:      connected,
	}
}

func (t *TUI) Run() error {
	t.clearScreen()
	t.renderHeader()

	// Show connection status via TUI
	t.printConnectionStatus()

	// Set up streaming callback - all output goes through TUI
	t.agent.SetStreamCallback(func(content string) {
		t.print(content)
	})

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("\n%s ", promptStyle.Render("➜"))
		if !scanner.Scan() {
			close(t.done)
			return nil
		}

		input := scanner.Text()
		t.handleInput(input)
	}
}

// print handles all output - formats and displays markdown
func (t *TUI) print(content string) {
	if content == "" {
		return
	}

	// Don't trim newlines - preserve them for proper streaming output
	cleanContent := strings.ReplaceAll(content, "\r\n", "\n")

	// For streaming output, just print directly without markdown rendering
	// The full response will be rendered when complete
	fmt.Print(cleanContent)
}

func (t *TUI) handleInput(input string) {
	switch input {
	case "exit", "quit":
		close(t.done)
	case "q":
		close(t.done)
	case "/help", "help":
		t.showHelp()
	case "/clear", "clear":
		t.clearScreen()
		t.renderHeader()
	case "/config", "config":
		t.showConfig()
	case "/style", "style":
		t.showStyles()
	default:
		if strings.TrimSpace(input) == "" {
			return
		}
		t.executeTask(input)
	}
}

func (t *TUI) executeTask(task string) {
	fmt.Println()

	ctx := context.Background()
	_, _ = t.agent.Execute(ctx, task)
}

func (t *TUI) showHelp() {
	t.print(`# Available Commands

- **/help** - Show this help message
- **/clear** - Clear the screen
- **/config** - Show current configuration  
- **/style** - Show available markdown styles
- **/exit** or **q** - Exit the application

## Available Tools

The agent can use these tools automatically:

- **read_file** - Read file contents
- **write_file** - Write content to files
- **run_command** - Execute shell commands
- **list_directory** - List directory contents
- **search_file** - Search for files by name
- **search_directory** - Search for directories
- **grep_search** - Search file contents
- **replace_file_content** - Find and replace in files
- **read_url** - Fetch web content
- **file_exists** - Check if file/directory exists
- **get_working_directory** - Get current directory
- **get_env_vars** - List environment variables
- **glob** - Find files by pattern
- **line_count** - Count lines in file
- **file_extension** - Get file extension

Type your request and press Enter to start!`)
}

func (t *TUI) showConfig() {
	t.print(fmt.Sprintf(`# Current Configuration

- **Provider:** %s
- **Model:** %s  
- **Endpoint:** %s
- **Context Window:** %d tokens
- **Temperature:** %.2f
- **Timeout:** %d seconds`,
		t.cfg.Provider, t.cfg.Model, t.cfg.LLMEndpoint,
		t.cfg.ContextWindow, t.cfg.Temperature, t.cfg.LLMTimeoutSeconds))
}

func (t *TUI) showStyles() {
	t.print(`# Available Markdown Styles

- dark - Dark theme (default)
- light - Light theme
- pink - Pink theme
- aurora - Aurora theme
- notty - Notty theme
- chocolate - Chocolate theme

Set the GLAMOUR_STYLE environment variable to change the default style.`)
}

func (t *TUI) clearScreen() {
	fmt.Print("\033[2J\033[H")
}

func (t *TUI) renderHeader() {
	headerBorder := lipgloss.NewStyle().
		Border(lipgloss.DoubleBorder()).
		BorderForeground(lipgloss.Color("238")).
		Padding(0, 1)

	headerContent := lipgloss.JoinVertical(
		lipgloss.Left,
		accentStyle.Render("  🤖 Agentic Coder")+dimStyle.Render(" - LM Studio Edition"),
		"",
		dimStyle.Render("  Provider: ")+accentStyle.Render(string(t.cfg.Provider))+dimStyle.Render("    Model: ")+accentStyle.Render(t.cfg.Model),
		dimStyle.Render("  Endpoint: ")+accentStyle.Render(t.cfg.LLMEndpoint),
	)

	fmt.Println(headerBorder.Render(headerContent))
	fmt.Println()

	helpText := dimStyle.Render("Commands: ") +
		accentStyle.Render("/help") + dimStyle.Render(" | ") +
		accentStyle.Render("/clear") + dimStyle.Render(" | ") +
		accentStyle.Render("/config") + dimStyle.Render(" | ") +
		accentStyle.Render("/style") + dimStyle.Render(" | ") +
		accentStyle.Render("/exit")

	fmt.Println(helpText)
	fmt.Println()
}

func (t *TUI) printConnectionStatus() {
	if t.connected {
		t.print("✅ Connected to " + string(t.cfg.Provider) + " at " + t.cfg.LLMEndpoint)
	} else {
		t.print("⚠️ Could not connect to " + string(t.cfg.Provider) + " at " + t.cfg.LLMEndpoint + " - continuing anyway")
	}
}
