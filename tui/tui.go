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
	colorReset  = "\033[0m"
	colorGreen  = "\033[32m"
	colorCyan   = "\033[36m"
	colorYellow = "\033[33m"
	colorRed    = "\033[31m"
	colorGrey   = "\033[90m"
	colorBold   = "\033[1m"
)

type TUI struct {
	agent   *orchestrator.Agent
	cfg     *config.AgentConfig
	history []Message
	done    chan struct{}
}

type Message struct {
	Role     string
	Content  string
	Time     time.Time
	ToolName string
}

func NewTUI(agent *orchestrator.Agent, cfg *config.AgentConfig) *TUI {
	return &TUI{
		agent:   agent,
		cfg:     cfg,
		history: []Message{},
		done:    make(chan struct{}),
	}
}

func (t *TUI) Run() error {
	t.renderHeader()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("\n%s👤 You:\033[0m ", colorGreen)
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
	case "/help":
		t.showHelp()
	case "/clear":
		t.clearScreen()
	case "/config":
		t.showConfig()
	default:
		if strings.TrimSpace(input) == "" {
			return
		}
		t.executeTask(input)
	}
}

func (t *TUI) executeTask(task string) {
	t.history = append(t.history, Message{Role: "user", Content: task, Time: time.Now()})

	ctx := context.Background()
	_, err := t.agent.Execute(ctx, task)
	
	if err != nil {
		t.history = append(t.history, Message{Role: "error", Content: err.Error(), Time: time.Now()})
	}
}

func (t *TUI) showHelp() {
	fmt.Printf("\n%s📖 Help - Available Commands:\n", colorYellow)
	fmt.Println("  • Type any task and press Enter to execute")
	fmt.Println("  • /help    - Show this help message")
	fmt.Println("  • /clear   - Clear the screen")
	fmt.Println("  • /config  - Show current configuration")
	fmt.Printf("%sTip: The agent uses tools automatically.%s\n", colorGrey, colorReset)
}

func (t *TUI) clearScreen() {
	t.history = nil
}

func (t *TUI) showConfig() {
	fmt.Printf("\n%s⚙️ Current Configuration:\n", colorYellow)
	fmt.Printf("  Provider:   %s\n", t.cfg.Provider)
	fmt.Printf("  Model:      %s\n", t.cfg.Model)
	fmt.Printf("  Endpoint:   %s\n", t.cfg.LLMEndpoint)
	fmt.Printf("  Context:    %d tokens\n", t.cfg.ContextWindow)
	fmt.Printf("  Temperature:%.2f\n", t.cfg.Temperature)
	fmt.Printf("%sTimeout:      %ds%s\n\n", colorGrey, t.cfg.LLMTimeoutSeconds, colorReset)
}

func (t *TUI) renderHeader() {
	fmt.Println(colorBold + "╔══════════════════════════════════════════════════════╗")
	fmt.Println("║    🤖 Agentic Coder - LM Studio Edition            ║")
	fmt.Println("╚══════════════════════════════════════════════════════╝" + colorReset)
	fmt.Printf("%sProvider:%s %s | %sModel:%s     %s\n", colorGrey, colorReset, t.cfg.Provider, colorReset, colorReset, t.cfg.Model)
	fmt.Printf("%sEndpoint:%s  %s\n", colorGrey, colorReset, t.cfg.LLMEndpoint)
	fmt.Println(colorGreen + "───────────────────────────────────────────────────────" + colorReset)
}
