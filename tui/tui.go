package tui

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
	"time"

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

type Message struct {
	Role      string
	Content   string
	Thinking  string
	Time      time.Time
	ToolName  string
	ToolArgs  string
	ToolResult string
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
	// Clear previous thinking
	t.agent.SetCurrentOutput("", "")
	
	fmt.Println()
	fmt.Printf("%s%s◌ Thinking...%s\n", Dim, Yellow, Reset)

	ctx := context.Background()
	_, err := t.agent.Execute(ctx, task)
	
	// Get the thinking and content from agent
	thinking, content := t.agent.GetCurrentOutput()
	
	// Display thinking if exists
	if thinking != "" {
		t.displayThinking(thinking)
	}
	
	// Display content
	if content != "" {
		t.displayMessage("assistant", content)
	} else if thinking == "" && content == "" {
		// Task completed without content (like a simple acknowledgment)
		fmt.Printf("\n%s%s✓ Done%s\n", Bold, Green, Reset)
	}
	
	if err != nil {
		t.displayMessage("error", err.Error())
	}
}

func (t *TUI) displayThinking(thinking string) {
	fmt.Printf("\n%s%s┌─ %s🤔 Thinking%s\n", Bold, Grey, Purple, Grey)
	lines := wrapText(thinking, 65)
	for _, line := range lines {
		fmt.Printf("%s%s│%s %s%s\n", Bold, Grey, Reset, Dim, line)
	}
	fmt.Printf("%s%s└%s\n", Bold, Grey, Reset)
}

func (t *TUI) displayMessage(role, content string) {
	var icon, iconColor, borderColor string
	
	switch role {
	case "user":
		icon, iconColor, borderColor = "👤", Green, Grey
	case "assistant":
		icon, iconColor, borderColor = "🤖", Cyan, Grey
	case "tool":
		icon, iconColor, borderColor = "🔧", Yellow, Grey
	case "error":
		icon, iconColor, borderColor = "✗", Red, Grey
	default:
		icon, iconColor, borderColor = "●", Cyan, Grey
	}
	
	if role == "error" {
		fmt.Printf("\n%s%s┌─ %sError%s\n", Bold, borderColor, iconColor, Grey)
	} else {
		fmt.Printf("\n%s%s┌─ %s%s %s%s\n", Bold, borderColor, iconColor, icon, Grey)
	}
	
	lines := wrapText(content, 65)
	for i, line := range lines {
		if i == 0 {
			fmt.Printf("%s%s│%s %s\n", Bold, borderColor, Reset, line)
		} else {
			fmt.Printf("%s%s │%s %s\n", Bold, borderColor, Reset, line)
		}
	}
	fmt.Printf("%s%s└%s\n", Bold, borderColor, Reset)
}

func wrapText(text string, width int) []string {
	if len(text) <= width {
		return []string{text}
	}
	
	var lines []string
	words := strings.Fields(text)
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

func (t *TUI) showHelp() {
	fmt.Printf("\n%s%s┌─ %s📖 Help%s\n", Bold, Grey, Yellow, Grey)
	fmt.Printf("%s%s│%s\n", Bold, Grey, Reset)
	fmt.Printf("%s%s│%s   %s/help%s      - Show this help\n", Bold, Grey, Reset, Green, Reset)
	fmt.Printf("%s%s│%s   %s/clear%s    - Clear screen\n", Bold, Grey, Reset, Green, Reset)
	fmt.Printf("%s%s│%s   %s/config%s    - Show configuration\n", Bold, Grey, Reset, Green, Reset)
	fmt.Printf("%s%s│%s   %s/exit%s      - Exit\n", Bold, Grey, Reset, Green, Reset)
	fmt.Printf("%s%s│%s   %s/q%s         - Exit (shortcut)\n", Bold, Grey, Reset, Green, Reset)
	fmt.Printf("%s%s│%s\n", Bold, Grey, Reset)
	fmt.Printf("%s%s│%s   Type your request and press Enter\n", Bold, Grey, Reset)
	fmt.Printf("%s%s│%s   The agent will use tools automatically\n", Bold, Grey, Reset)
	fmt.Printf("%s%s└%s\n", Bold, Grey, Reset)
}

func (t *TUI) clearScreen() {
	fmt.Print("\033[2J\033[H")
}

func (t *TUI) showConfig() {
	fmt.Printf("\n%s%s┌─ %s⚙️ Configuration%s\n", Bold, Grey, Yellow, Grey)
	fmt.Printf("%s%s│%s\n", Bold, Grey, Reset)
	fmt.Printf("%s%s│%s   %sProvider:%s   %s%s\n", Bold, Grey, Reset, Dim, Reset, Cyan, t.cfg.Provider)
	fmt.Printf("%s%s│%s   %sModel:%s      %s%s\n", Bold, Grey, Reset, Dim, Reset, Cyan, t.cfg.Model)
	fmt.Printf("%s%s│%s   %sEndpoint:%s   %s%s\n", Bold, Grey, Reset, Dim, Reset, Cyan, t.cfg.LLMEndpoint)
	fmt.Printf("%s%s│%s   %sContext:%s    %s%d tokens\n", Bold, Grey, Reset, Dim, Reset, Cyan, t.cfg.ContextWindow)
	fmt.Printf("%s%s│%s   %sTemperature:%s %.2f\n", Bold, Grey, Reset, Dim, Reset, Cyan, t.cfg.Temperature)
	fmt.Printf("%s%s│%s   %sTimeout:%s     %ds\n", Bold, Grey, Reset, Dim, Reset, Cyan, t.cfg.LLMTimeoutSeconds)
	fmt.Printf("%s%s│%s\n", Bold, Grey, Reset)
	fmt.Printf("%s%s└%s\n", Bold, Grey, Reset)
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
