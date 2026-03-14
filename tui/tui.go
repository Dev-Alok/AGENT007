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
)

type TUI struct {
	agent          *orchestrator.Agent
	cfg            *config.AgentConfig
	done           chan struct{}
	markdownRender *renderer.MarkdownRenderer
	connected      bool
	currentTask    string
	lastStatus     string
	taskStartTime  time.Time
	iteration      int
	maxIterations  int
}

func NewTUI(agent *orchestrator.Agent, cfg *config.AgentConfig, connected bool) *TUI {
	mr, _ := renderer.NewAutoStyleRenderer()
	tui := &TUI{
		agent:          agent,
		cfg:            cfg,
		done:           make(chan struct{}),
		markdownRender: mr,
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
		t.output(content)
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
		t.lastStatus = event.Message
		if strings.Contains(event.Message, "Starting") {
			t.taskStartTime = time.Now()
		}

	case orchestrator.EventIteration:
		var meta orchestrator.IterationMeta
		if err := json.Unmarshal(event.Metadata, &meta); err == nil {
			t.iteration = meta.Current
			t.maxIterations = meta.Max
			t.lastStatus = formatIteration(meta.Current, meta.Max)
			t.renderStatusBar()
		}

	case orchestrator.EventPlanning:
		if strings.HasPrefix(event.Message, "Task:") {
			t.output(formatTag("TASK", event.Message))
		} else if strings.HasPrefix(event.Message, "Executing") {
			t.output(formatTag("TOOLS", event.Message))
		} else {
			t.output(formatTag("PLAN", event.Message))
		}

	case orchestrator.EventThinking:
		msg := event.Message
		if strings.HasPrefix(msg, "Reasoning: ") {
			msg = strings.TrimPrefix(msg, "Reasoning: ")
		}
		t.lastStatus = "Thinking..."
		t.output(formatSubtle("THINK", msg))

	case orchestrator.EventToolCall:
		var meta orchestrator.ToolCallMeta
		if err := json.Unmarshal(event.Metadata, &meta); err == nil {
			argsStr := formatArgs(meta.Args)
			if argsStr != "" {
				t.output(formatCall(meta.ToolName, argsStr))
			} else {
				t.output(formatCallNoArgs(meta.ToolName))
			}
		} else {
			t.output(formatSubtle("CALL", event.Message))
		}

	case orchestrator.EventToolResult:
		t.lastStatus = event.Message
		t.output(formatSubtle("OK", event.Message))

	case orchestrator.EventToolError:
		t.output(formatSubtle("ERR", event.Message))

	case orchestrator.EventRetry:
		var meta orchestrator.RetryMeta
		if err := json.Unmarshal(event.Metadata, &meta); err == nil {
			t.output(formatRetry(meta.ToolName, meta.Attempt, meta.MaxAttempts))
			if meta.Error != "" {
				t.output(formatError(meta.Error))
			}
		} else {
			t.output(formatSubtle("RETRY", event.Message))
		}

	case orchestrator.EventComplete:
		elapsed := time.Since(t.taskStartTime)
		t.output(formatDone(elapsed.Round(time.Second)))
		t.lastStatus = "Ready"
		t.renderStatusBar()

	case orchestrator.EventStream:
		t.output(event.Message)
	}
}

func (t *TUI) renderStatusBar() {
	border := lipgloss.NewStyle().
		Border(lipgloss.NormalBorder()).
		BorderForeground(lipgloss.Color("238")).
		Padding(0, 1)

	statusContent := lipgloss.JoinHorizontal(
		lipgloss.Left,
		dimStyle.Render("Status: "),
		accentStyle.Render(t.lastStatus),
		dimStyle.Render(" | "),
		dimStyle.Render("Task: "),
		accentStyle.Render(truncateString(t.currentTask, 40)),
	)

	t.output("\n")
	t.output(border.Render(statusContent))
	t.output("\n")
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
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
	t.lastStatus = "Starting..."
	t.output("\n[START] Executing task...")

	ctx := context.Background()
	_, _ = t.agent.Execute(ctx, task)
}

func (t *TUI) showStatus() {
	elapsed := time.Since(t.taskStartTime)
	statusMsg := formatStatus(t.currentTask, elapsed.Round(time.Second), t.iteration, t.maxIterations, t.lastStatus)
	t.renderMarkdown(statusMsg)
}

func (t *TUI) showHelp() {
	t.renderMarkdown("# Available Commands\n\n- /help - Show this help message\n- /clear - Clear the screen\n- /config - Show current configuration\n- /style - Show available markdown styles\n- /status - Show current task status\n- /exit or q - Exit the application\n\n## Agent Events\n\nThe agent will notify you about:\n- [TASK] Task planning\n- [THINK] Thinking/reasoning\n- [TOOLS] Tool execution\n- [OK] Tool results\n- [ERR] Tool errors\n- [RETRY] Retries")
}

func (t *TUI) showConfig() {
	t.renderMarkdown(formatConfig(string(t.cfg.Provider), t.cfg.Model, t.cfg.LLMEndpoint, t.cfg.ContextWindow, float64(t.cfg.Temperature), t.cfg.LLMTimeoutSeconds))
}

func (t *TUI) showStyles() {
	t.renderMarkdown("# Available Markdown Styles\n\n- dark - Dark theme (default)\n- light - Light theme\n- pink - Pink theme\n- aurora - Aurora theme\n- notty - Notty theme\n- chocolate - Chocolate theme\n\nSet the GLAMOUR_STYLE environment variable to change the default style.")
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
		t.output("[OK] Connected to " + string(t.cfg.Provider) + " at " + t.cfg.LLMEndpoint)
	} else {
		t.output("[WARN] Could not connect to " + string(t.cfg.Provider) + " at " + t.cfg.LLMEndpoint + " - continuing anyway")
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
	rendered, err := t.markdownRender.Render(content)
	if err != nil || rendered == content {
		t.output(content)
		return
	}
	t.output(rendered)
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

func formatTag(tag, msg string) string {
	return "\n[" + tag + "] " + msg
}

func formatSubtle(tag, msg string) string {
	return "  [" + tag + "] " + msg
}

func formatCall(name, args string) string {
	return "  [CALL] " + name + "(" + args + ")"
}

func formatCallNoArgs(name string) string {
	return "  [CALL] " + name
}

func formatRetry(name string, attempt, max int) string {
	return "  [RETRY] " + name + " (attempt " + formatInt(attempt) + "/" + formatInt(max) + ")"
}

func formatError(err string) string {
	return "    Error: " + err
}

func formatDone(d time.Duration) string {
	return "\n[DONE] Task completed in " + d.String()
}

func formatIteration(current, max int) string {
	return "Iteration " + formatInt(current) + "/" + formatInt(max)
}

func formatStatus(task string, elapsed time.Duration, iter, maxIter int, status string) string {
	var b strings.Builder
	b.WriteString("# Current Status\n\n")
	b.WriteString("- Task: ")
	b.WriteString(task)
	b.WriteString("\n- Elapsed: ")
	b.WriteString(elapsed.String())
	b.WriteString("\n- Iteration: ")
	b.WriteString(formatInt(iter))
	b.WriteString("/")
	b.WriteString(formatInt(maxIter))
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
	b.WriteString(formatInt(ctxWindow))
	b.WriteString(" tokens\n- Temperature: ")
	b.WriteString(formatFloat(temp))
	b.WriteString("\n- Timeout: ")
	b.WriteString(formatInt(timeout))
	b.WriteString(" seconds")
	return b.String()
}

func formatInt(n int) string {
	return strings.TrimSpace(strconv.Itoa(n))
}

func formatFloat(f float64) string {
	return strings.TrimSpace(strconv.FormatFloat(f, 'f', 2, 64))
}
