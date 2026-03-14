package tui

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"agentic-coder/pkg/config"
	"agentic-coder/pkg/orchestrator"
)

const (
	Reset  = "\033[0m"
	Bold   = "\033[1m"
	Dim    = "\033[2m"

	Green  = "\033[32m"
	Yellow = "\033[33m"
	Cyan   = "\033[36m"
	Grey   = "\033[90m"
	Red    = "\033[31m"
	Purple = "\033[35m"
)

type TUI struct {
	agent   *orchestrator.Agent
	cfg     *config.AgentConfig
	done    chan struct{}
}

func NewTUI(agent *orchestrator.Agent, cfg *config.AgentConfig) *TUI {
	return &TUI{
		agent: agent,
		cfg:   cfg,
		done:  make(chan struct{}),
	}
}

func (t *TUI) Run() error {
	t.clearScreen()
	t.renderHeader()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("\n%s%s➜%s ", Bold, Green, Reset)
		if !scanner.Scan() {
			close(t.done)
			return nil
		}

		input := scanner.Text()
		t.handleInput(input)
	}
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
	_, err := t.agent.Execute(ctx, task)
	
	// Get the thinking and content from agent
	thinking, content := t.agent.GetCurrentOutput()
	
	// Display thinking if exists and is different from content
	if thinking != "" && thinking != content {
		t.displayBlock("🤔 Thinking", thinking, Purple)
		fmt.Println()
	}
	
	// Display content if exists
	if content != "" {
		t.displayBlock("🤖 Assistant", content, Cyan)
	}
	
	if err != nil {
		t.displayBlock("✗ Error", err.Error(), Red)
	}
}

func (t *TUI) displayBlock(title, content string, titleColor string) {
	borderColor := Grey
	
	// First check if content was already printed (starts with common prefixes)
	if strings.HasPrefix(content, "\n\033[36m🤖 Assistant:\033[0m\n") {
		// Content was already printed by orchestrator, just show thinking part
		content = strings.TrimPrefix(content, "\n\033[36m🤖 Assistant:\033[0m\n")
	}
	
	// Remove any existing ANSI escape sequences for cleaner display
	// This is a simplified approach - we just display what we have
	
	lines := wrapText(content, 62)
	
	// Draw top border
	fmt.Printf("%s┌", borderColor)
	for i := 0; i < 66; i++ {
		fmt.Printf("─")
	}
	fmt.Printf("┐%s\n", Reset)
	
	// Title
	fmt.Printf("%s│%s %s %s%s\n", borderColor, Reset, titleColor, title, Reset)
	
	// Separator
	fmt.Printf("%s│", borderColor)
	for i := 0; i < 66; i++ {
		fmt.Printf("─")
	}
	fmt.Printf("│%s\n", Reset)
	
	// Content
	for i, line := range lines {
		padding := 66 - len(line)
		if i == len(lines)-1 {
			// Last line - use └┘
			fmt.Printf("%s│%s %s%s%s%s└%s\n", borderColor, Reset, line, strings.Repeat(" ", padding), borderColor, Reset)
		} else {
			fmt.Printf("%s│%s %s%s%s│%s\n", borderColor, Reset, line, strings.Repeat(" ", padding), borderColor, Reset)
		}
	}
	
	// Bottom border
	fmt.Printf("%s└", borderColor)
	for i := 0; i < 66; i++ {
		fmt.Printf("─")
	}
	fmt.Printf("┘%s", Reset)
}

func wrapText(text string, width int) []string {
	// First clean the text of ANSI codes for proper wrapping
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
			// Skip until we find the letter at the end of escape sequence
			for i < len(s) && s[i] != 'm' {
				i++
			}
			continue
		}
		result.WriteByte(s[i])
	}
	return result.String()
}

func (t *TUI) showHelp() {
	t.displayBlock("📖 Help", `Commands:
  /help      - Show this help
  /clear     - Clear screen  
  /config    - Show configuration
  /exit      - Exit
  q          - Exit (shortcut)

Type your request and press Enter.
The agent will use tools automatically.`, Yellow)
}

func (t *TUI) clearScreen() {
	fmt.Print("\033[2J\033[H")
}

func (t *TUI) showConfig() {
	info := fmt.Sprintf(`Provider:    %s
Model:       %s
Endpoint:    %s
Context:     %d tokens
Temperature: %.2f
Timeout:     %d seconds`, 
		t.cfg.Provider, t.cfg.Model, t.cfg.LLMEndpoint, 
		t.cfg.ContextWindow, t.cfg.Temperature, t.cfg.LLMTimeoutSeconds)
	
	t.displayBlock("⚙️ Configuration", info, Yellow)
}

func (t *TUI) renderHeader() {
	borderColor := Grey
	accentColor := Cyan
	
	fmt.Printf("%s╔═══════════════════════════════════════════════════════════════════╗%s\n", borderColor, Reset)
	fmt.Printf("%s║%s  %s🤖 Agentic Coder%s                                             %s║%s\n", borderColor, Reset, Bold+accentColor, Reset, borderColor, Reset)
	fmt.Printf("%s║%s  %sLM Studio Edition%s                                         %s║%s\n", borderColor, Reset, Dim, Reset, borderColor, Reset)
	fmt.Printf("%s╠═══════════════════════════════════════════════════════════════════╣%s\n", borderColor, Reset)
	fmt.Printf("%s║%s  %sProvider:%s  %s%-12s  %sModel:%s     %s%-25s %s║%s\n", 
		borderColor, Reset, Dim, Reset, Cyan, t.cfg.Provider, Dim, Reset, Cyan, t.cfg.Model, borderColor, Reset)
	fmt.Printf("%s║%s  %sEndpoint:%s %s%-45s %s║%s\n", 
		borderColor, Reset, Dim, Reset, Cyan, t.cfg.LLMEndpoint, borderColor, Reset)
	fmt.Printf("%s╚═══════════════════════════════════════════════════════════════════╝%s\n", borderColor, Reset)
	
	fmt.Printf("\n%s%sCommands:%s /help %s| %s/clear %s| %s/config %s| %s/exit%s\n\n", 
		Dim, Reset, Green, Dim, Reset, Green, Dim, Reset, Green, Reset)
}
