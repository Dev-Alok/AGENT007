package tui

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"sync"
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
	agent      *orchestrator.Agent
	cfg        *config.AgentConfig
	mu         sync.Mutex
	history    []Message
	done       chan struct{}
}

type Message struct {
	Role    string
	Content string
	Time    time.Time
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
	go t.renderer()

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
		go t.executeTask(input)
	}
}

func (t *TUI) executeTask(task string) {
	t.mu.Lock()
	t.history = append(t.history, Message{Role: "user", Content: task, Time: time.Now()})
	t.mu.Unlock()

	ctx := context.Background()
	_, err := t.agent.Execute(ctx, task)
	if err != nil {
		t.mu.Lock()
		t.history = append(t.history, Message{Role: "error", Content: err.Error(), Time: time.Now()})
		t.mu.Unlock()
	}
}

func (t *TUI) renderer() {
	for {
		select {
		case <-t.done:
			return
		default:
			t.mu.Lock()
			historyCopy := make([]Message, len(t.history))
			copy(historyCopy, t.history)
			t.mu.Unlock()

			fmt.Print("\033[H\033[J")
			t.renderHeader()
			
			for _, msg := range historyCopy {
				switch msg.Role {
				case "user":
					fmt.Printf("\n%s👤 You:\033[0m %s\n", colorGreen, msg.Content)
				case "assistant":
					fmt.Printf("\n%s🤖 Assistant:\n%s%s\n", colorCyan, colorReset, msg.Content)
				case "tool":
					fmt.Printf("\n%s⚙️ Tool: %s\033[0m\n", colorYellow, msg.ToolName)
					fmt.Printf("%s  Output:\n%s%s\n", colorGrey, colorReset, msg.Content)
				case "error":
					fmt.Printf("\n%s❌ Error:\033[0m %s\n", colorRed, msg.Content)
				}
			}

			time.Sleep(50 * time.Millisecond)
		}
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
	t.mu.Lock()
	t.history = nil
	t.mu.Unlock()
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
