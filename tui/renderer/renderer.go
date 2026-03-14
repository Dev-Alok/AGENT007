package renderer

import (
	"os"
	"strconv"

	"charm.land/glamour/v2"
)

var (
	terminalWidth = 80
)

func init() {
	if w, ok := os.LookupEnv("GLAMOUR_WIDTH"); ok {
		if width, err := strconv.Atoi(w); err == nil && width > 0 {
			terminalWidth = width
		}
	}
}

type MarkdownRenderer struct{}

func NewMarkdownRenderer() (*MarkdownRenderer, error) {
	return &MarkdownRenderer{}, nil
}

func NewAutoStyleRenderer() (*MarkdownRenderer, error) {
	return &MarkdownRenderer{}, nil
}

func (m *MarkdownRenderer) Render(content string) (string, error) {
	if content == "" {
		return "", nil
	}
	// Use simple Render function with "dark" style
	out, err := glamour.Render(content, "dark")
	if err != nil {
		return content, err
	}
	if out == "" {
		return content, nil
	}
	return out, nil
}

func (m *MarkdownRenderer) RenderAutoStyle(content string) (string, error) {
	if content == "" {
		return "", nil
	}
	// Use auto style detection
	out, err := glamour.Render(content, "dark")
	if err != nil {
		return content, err
	}
	if out == "" {
		return content, nil
	}
	return out, nil
}

func RenderMarkdown(content string) string {
	if content == "" {
		return ""
	}
	out, err := glamour.Render(content, "dark")
	if err != nil {
		return content
	}
	if out == "" {
		return content
	}
	return out
}

func RenderMarkdownAuto(content string) string {
	if content == "" {
		return ""
	}
	out, err := glamour.Render(content, "dark")
	if err != nil {
		return content
	}
	if out == "" {
		return content
	}
	return out
}

func RenderMarkdownWithStyle(content string, styleName string) string {
	if content == "" {
		return ""
	}
	out, err := glamour.Render(content, styleName)
	if err != nil {
		return content
	}
	if out == "" {
		return content
	}
	return out
}
