package tui

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"time"

	"agentic-coder/pkg/config"
	"agentic-coder/pkg/orchestrator"
	"agentic-coder/tui/renderer"
	"charm.land/lipgloss/v2"
	md "github.com/MichaelMure/go-term-markdown"
)

var (
	accentStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("86"))
	dimStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	promptStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("154")).Bold(true)
	successStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("76"))
	warningStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("220"))
	errorStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
	infoStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("39"))
	thinkingStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("98"))
	boldStyle     = lipgloss.NewStyle().Bold(true)
	mutedStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
)

type TUI struct {
	agent            *orchestrator.Agent
	cfg              *config.AgentConfig
	done             chan struct{}
	streamRenderer   *renderer.StreamingRenderer
	connected        bool
	currentTask      string
	lastStatus       string
	taskStartTime    time.Time
	iteration        int
	maxIterations    int
	toolsUsed        bool
	toolResultBuffer strings.Builder
	inToolResult     bool
	lastStatusBar    string
}

func NewTUI(agent *orchestrator.Agent, cfg *config.AgentConfig, connected bool) *TUI {
	tui := &TUI{
		agent:          agent,
		cfg:            cfg,
		done:           make(chan struct{}),
		streamRenderer: renderer.NewStreamingRenderer(),
		connected:      connected,
		maxIterations:  cfg.MaxIterations,
	}
	if tui.maxIterations <= 0 {
		tui.maxIterations = 20
	}
	return tui
}

func (t *TUI) Run() error {
	t.clearScreen()
	t.renderHeader()
	t.printConnectionStatus()

	t.agent.SetStreamCallback(func(content string) {
		t.streamOutput(content)
	})

	t.agent.SetEventCallback(func(event orchestrator.AgentEvent) {
		t.handleEvent(event)
	})

	scanner := bufio.NewScanner(os.Stdin)
	for {
		t.output("\n")
		t.renderPrompt()
		if !scanner.Scan() {
			close(t.done)
			return nil
		}

		input := scanner.Text()
		t.handleInput(input)
	}
}

func (t *TUI) renderPrompt() {
	t.output(promptStyle.Render("> "))
}

func (t *TUI) handleEvent(event orchestrator.AgentEvent) {
	switch event.Type {
	case orchestrator.EventStatus:
		if strings.Contains(event.Message, "Starting") {
			t.taskStartTime = time.Now()
			t.currentTask = ""
			t.iteration = 0
			t.toolsUsed = false
			t.lastStatus = "Starting"
			t.toolResultBuffer.Reset()
			t.inToolResult = false
			t.lastStatusBar = ""
			t.output(formatStart())
		}

	case orchestrator.EventIteration:
		var meta orchestrator.IterationMeta
		if err := json.Unmarshal(event.Metadata, &meta); err == nil {
			t.iteration = meta.Current
			t.maxIterations = meta.Max
			t.lastStatus = formatIter(meta.Current, meta.Max)
			t.renderStatusBar()
		}

	case orchestrator.EventPlanning:
		if strings.HasPrefix(event.Message, "Executing") {
			if !t.toolsUsed {
				t.toolsUsed = true
			}
			t.output(formatTools(event.Message))
		}

	case orchestrator.EventThinking:
		msg := event.Message
		if strings.HasPrefix(msg, "Reasoning: ") {
			msg = strings.TrimPrefix(msg, "Reasoning: ")
			t.lastStatus = "Reasoning"
			t.output(formatThinking(msg))
		} else if msg == "Analyzing tool results..." {
			t.lastStatus = "Analyzing"
			t.output(formatThink(msg))
		}

	case orchestrator.EventToolCall:
		t.inToolResult = true
		t.toolResultBuffer.Reset()
		var meta orchestrator.ToolCallMeta
		if err := json.Unmarshal(event.Metadata, &meta); err == nil {
			argsStr := formatArgs(meta.Args)
			if argsStr != "" {
				t.output(formatCall(meta.ToolName, argsStr))
			} else {
				t.output(formatCallNoArgs(meta.ToolName))
			}
		} else {
			t.output(formatCallMsg(event.Message))
		}

	case orchestrator.EventToolResult:
		t.lastStatus = "Working"
		t.inToolResult = false
		t.output(formatResult(event.Message))

	case orchestrator.EventToolError:
		t.inToolResult = false
		t.output(formatError(event.Message))

	case orchestrator.EventRetry:
		var meta orchestrator.RetryMeta
		if err := json.Unmarshal(event.Metadata, &meta); err == nil {
			t.output(formatRetry(meta.ToolName, meta.Attempt, meta.MaxAttempts))
			if meta.Error != "" {
				t.output(formatErrorMsg(meta.Error))
			}
		} else {
			t.output(formatRetryMsg(event.Message))
		}

	case orchestrator.EventComplete:
		elapsed := time.Since(t.taskStartTime)
		if t.toolsUsed {
			t.output(formatDone(elapsed))
		} else {
			t.lastStatus = "Ready"
		}

	case orchestrator.EventStream:
	}
}

func (t *TUI) streamOutput(content string) {
	if content == "" {
		return
	}

	if t.inToolResult {
		t.toolResultBuffer.WriteString(content)
		os.Stdout.WriteString(content)
	} else {
		t.streamRenderer.Stream(content)
	}
}

func (t *TUI) renderStatusBar() {
	border := lipgloss.NewStyle().
		Border(lipgloss.NormalBorder()).
		BorderForeground(lipgloss.Color("238")).
		Padding(0, 1)

	elapsed := time.Since(t.taskStartTime)
	elapsedStr := formatDuration(elapsed)

	statusContent := lipgloss.JoinHorizontal(
		lipgloss.Left,
		dimStyle.Render("["),
		accentStyle.Render(t.lastStatus),
		dimStyle.Render("] "),
		dimStyle.Render("Elapsed: "),
		accentStyle.Render(elapsedStr),
		dimStyle.Render(" | "),
		dimStyle.Render("Iter: "),
		accentStyle.Render(strconv.Itoa(t.iteration)+"/"+strconv.Itoa(t.maxIterations)),
	)

	newBar := border.Render(statusContent)
	if newBar != t.lastStatusBar {
		t.lastStatusBar = newBar
		t.output("\n")
		t.output(newBar)
	}
}

func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return d.Round(time.Second).String()
	}
	if d < time.Hour {
		return d.Round(time.Second).String()
	}
	return d.Round(time.Minute).String()
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
	case "/status":
		t.showStatus()
	default:
		if strings.TrimSpace(input) == "" {
			return
		}
		t.executeTask(input)
	}
}

func (t *TUI) executeTask(task string) {
	t.currentTask = task
	t.taskStartTime = time.Now()
	t.iteration = 0
	t.toolsUsed = false
	t.lastStatus = "Starting"
	t.toolResultBuffer.Reset()
	t.inToolResult = false
	t.streamRenderer.Reset()
	t.lastStatusBar = ""

	ctx := context.Background()
	_, _ = t.agent.Execute(ctx, task)
}

func (t *TUI) showStatus() {
	elapsed := time.Since(t.taskStartTime)
	statusMsg := formatStatus(t.currentTask, elapsed, t.iteration, t.maxIterations, t.lastStatus)
	t.renderMarkdown(statusMsg)
}

func (t *TUI) showHelp() {
	t.renderMarkdown("# Available Commands\n\n- /help - Show this help message\n- /clear - Clear the screen\n- /config - Show current configuration\n- /style - Show available markdown styles\n- /status - Show current task status\n- /exit or q - Exit the application")
}

func (t *TUI) showConfig() {
	t.renderMarkdown(formatConfig(string(t.cfg.Provider), t.cfg.Model, t.cfg.LLMEndpoint, t.cfg.ContextWindow, float64(t.cfg.Temperature), t.cfg.LLMTimeoutSeconds))
}

func (t *TUI) showStyles() {
	t.renderMarkdown("# Available Markdown Styles\n\n- dark - Dark theme (default)\n- light - Light theme\n- pink - Pink theme\n- aurora - Aurora theme\n- notty - Notty theme\n- chocolate - Chocolate theme")
}

func (t *TUI) clearScreen() {
	t.output("\033[2J\033[H")
}

func (t *TUI) renderHeader() {
	border := lipgloss.NewStyle().
		Border(lipgloss.DoubleBorder()).
		BorderForeground(lipgloss.Color("238")).
		Padding(0, 1)

	content := lipgloss.JoinVertical(
		lipgloss.Left,
		accentStyle.Render("  Agentic Coder")+dimStyle.Render(" - LM Studio Edition"),
		"",
		dimStyle.Render("  Provider: ")+accentStyle.Render(string(t.cfg.Provider))+dimStyle.Render("    Model: ")+accentStyle.Render(t.cfg.Model),
		dimStyle.Render("  Endpoint: ")+accentStyle.Render(t.cfg.LLMEndpoint),
	)

	t.output(border.Render(content))
	t.output("\n")

	helpText := dimStyle.Render("Commands: ") +
		accentStyle.Render("/help") + dimStyle.Render(" | ") +
		accentStyle.Render("/clear") + dimStyle.Render(" | ") +
		accentStyle.Render("/config") + dimStyle.Render(" | ") +
		accentStyle.Render("/style") + dimStyle.Render(" | ") +
		accentStyle.Render("/status") + dimStyle.Render(" | ") +
		accentStyle.Render("/exit")

	t.output(helpText)
	t.output("\n")
}

func (t *TUI) printConnectionStatus() {
	if t.connected {
		t.output(successStyle.Render("✓") + " " + dimStyle.Render("Connected to ") + accentStyle.Render(string(t.cfg.Provider)) + dimStyle.Render(" at ") + mutedStyle.Render(t.cfg.LLMEndpoint))
	} else {
		t.output(warningStyle.Render("⚠") + " " + dimStyle.Render("Could not connect to ") + accentStyle.Render(string(t.cfg.Provider)))
	}
}

func (t *TUI) output(content string) {
	if content == "" {
		return
	}
	cleanContent := strings.ReplaceAll(content, "\r\n", "\n")
	os.Stdout.WriteString(cleanContent)
}

func (t *TUI) renderMarkdown(content string) {
	if content == "" {
		return
	}
	rendered := md.Render(content, 80, 0)
	os.Stdout.Write(rendered)
}

func formatArgs(args map[string]interface{}) string {
	if args == nil || len(args) == 0 {
		return ""
	}
	parts := make([]string, 0, len(args))
	for k, v := range args {
		parts = append(parts, k+"="+formatValue(v))
	}
	return strings.Join(parts, ", ")
}

func formatValue(v interface{}) string {
	switch val := v.(type) {
	case string:
		return val
	case int:
		return strconv.Itoa(val)
	case int64:
		return strconv.FormatInt(val, 10)
	case float64:
		return strconv.FormatFloat(val, 'f', -1, 64)
	case float32:
		return strconv.FormatFloat(float64(val), 'f', -1, 64)
	case bool:
		return strconv.FormatBool(val)
	default:
		b, _ := json.Marshal(val)
		return string(b)
	}
}

func formatStart() string {
	return "\n" + dimStyle.Render("[") + accentStyle.Render("START") + dimStyle.Render("] ") + dimStyle.Render("Executing task...\n")
}

func formatIter(current, max int) string {
	return "Iter " + strconv.Itoa(current) + "/" + strconv.Itoa(max)
}

func formatTools(msg string) string {
	return "\n" + dimStyle.Render("→") + " " + accentStyle.Render(msg) + "\n"
}

func formatThink(msg string) string {
	return dimStyle.Render("  ○ ") + thinkingStyle.Render(msg)
}

func formatThinking(msg string) string {
	return dimStyle.Render("  ◐ ") + thinkingStyle.Render(msg) + "\n"
}

func formatCall(name, args string) string {
	return dimStyle.Render("  ├─ ") + accentStyle.Render(name) + dimStyle.Render("("+args+")") + "\n"
}

func formatCallNoArgs(name string) string {
	return dimStyle.Render("  ├─ ") + accentStyle.Render(name) + "\n"
}

func formatCallMsg(msg string) string {
	return dimStyle.Render("  ├─ ") + msg + "\n"
}

func formatResult(msg string) string {
	return dimStyle.Render("  └─ ") + successStyle.Render("✓") + " " + dimStyle.Render(msg) + "\n"
}

func formatError(msg string) string {
	return dimStyle.Render("  └─ ") + errorStyle.Render("✗") + " " + errorStyle.Render(msg) + "\n"
}

func formatErrorMsg(err string) string {
	return "    " + dimStyle.Render("Error: ") + err + "\n"
}

func formatRetry(name string, attempt, max int) string {
	return dimStyle.Render("  └─ ") + warningStyle.Render("↻") + " " + dimStyle.Render(name+" (attempt "+strconv.Itoa(attempt)+"/"+strconv.Itoa(max)+")") + "\n"
}

func formatRetryMsg(msg string) string {
	return dimStyle.Render("  └─ ") + warningStyle.Render("↻") + " " + msg + "\n"
}

func formatDone(d time.Duration) string {
	return "\n" + dimStyle.Render("[") + successStyle.Render("DONE") + dimStyle.Render("] ") + dimStyle.Render("Completed in "+formatDuration(d)) + "\n"
}

func formatStatus(task string, elapsed time.Duration, iter, maxIter int, status string) string {
	var b strings.Builder
	b.WriteString("# Current Status\n\n")
	b.WriteString("- Task: ")
	b.WriteString(task)
	b.WriteString("\n- Elapsed: ")
	b.WriteString(formatDuration(elapsed))
	b.WriteString("\n- Iteration: ")
	b.WriteString(strconv.Itoa(iter))
	b.WriteString("/")
	b.WriteString(strconv.Itoa(maxIter))
	b.WriteString("\n- Status: ")
	b.WriteString(status)
	return b.String()
}

func formatConfig(provider, model, endpoint string, ctxWindow int, temp float64, timeout int) string {
	var b strings.Builder
	b.WriteString("# Current Configuration\n\n")
	b.WriteString("- Provider: ")
	b.WriteString(provider)
	b.WriteString("\n- Model: ")
	b.WriteString(model)
	b.WriteString("\n- Endpoint: ")
	b.WriteString(endpoint)
	b.WriteString("\n- Context Window: ")
	b.WriteString(strconv.Itoa(ctxWindow))
	b.WriteString(" tokens\n- Temperature: ")
	b.WriteString(strconv.FormatFloat(temp, 'f', 2, 64))
	b.WriteString("\n- Timeout: ")
	b.WriteString(strconv.Itoa(timeout))
	b.WriteString(" seconds")
	return b.String()
}
