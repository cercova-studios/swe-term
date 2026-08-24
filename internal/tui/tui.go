package tui

import (
	"context"
	"fmt"
	"strings"
	"time"

	"charm.land/bubbles/v2/textinput"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/glamour/v2"
	"charm.land/lipgloss/v2"

	"swe-term/internal/config"
	"swe-term/internal/core"
)

const helpText = `slash commands
  /config   print user and project config.toml
  /help     this text
  /quit     exit

Anything else is sent to the model. ctrl+c also quits.`

var (
	inputBoxStyle = lipgloss.NewStyle().
			Border(lipgloss.NormalBorder()).
			BorderForeground(lipgloss.Color("240")).
			Padding(0, 1)
	slashMenuStyle = lipgloss.NewStyle().
			Border(lipgloss.NormalBorder()).
			BorderForeground(lipgloss.Color("240")).
			Padding(0, 1)
	slashItemStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
	slashSelStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("230")).Background(lipgloss.Color("238"))
	statusStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
	workingStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("250"))
	spinFrames     = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
)

type Options struct {
	Result      config.Result
	Provider    core.Provider // nil if construction failed
	ProviderErr error
	LoadArgs    []string
	NewProvider func(config.Config) (core.Provider, error)
}

type model struct {
	viewport viewport.Model
	input    textinput.Model
	lines    []string
	width    int
	height   int
	ready    bool

	cfg         config.Config
	userPath    string
	projectPath string
	loadArgs    []string
	provider    core.Provider
	providerErr error
	newProvider func(config.Config) (core.Provider, error)
	busy        bool
	history     []core.Message

	slashHits []slashCmd
	slashSel  int

	streamID        uint64
	streamCh        <-chan core.StreamEvent
	streamCancel    context.CancelFunc
	streamRaw       string
	streamLine      int
	streamCompleted bool
	lastUsage       core.Usage
	sessionUsage    core.UsageTotals

	follow      bool
	workStarted time.Time
	spin        int
}

type tickMsg time.Time
type streamStartMsg struct {
	id  uint64
	ch  <-chan core.StreamEvent
	err error
}

type streamDeltaMsg struct {
	id uint64
	ev core.StreamEvent
	ch <-chan core.StreamEvent
}

type streamDoneMsg struct {
	id uint64
}

func Run(opts Options) error {
	m := newModel(opts)
	_, err := tea.NewProgram(m).Run()
	return err
}

func newModel(opts Options) model {
	in := textinput.New()
	in.Prompt = "> "
	in.SetVirtualCursor(false)
	in.CharLimit = 0
	in.Focus()

	s := textinput.DefaultDarkStyles()
	s.Cursor.Shape = tea.CursorBlock
	in.SetStyles(s)

	vp := viewport.New()
	vp.KeyMap.Left.SetEnabled(false)
	vp.KeyMap.Right.SetEnabled(false)

	return model{
		viewport:    vp,
		input:       in,
		cfg:         opts.Result.Config,
		userPath:    opts.Result.UserPath,
		projectPath: opts.Result.ProjectPath,
		loadArgs:    opts.LoadArgs,
		provider:    opts.Provider,
		providerErr: opts.ProviderErr,
		newProvider: opts.NewProvider,
		streamLine:  -1,
		follow:      true,
	}
}

func (m model) Init() tea.Cmd {
	return textinput.Blink
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		inner := max(1, msg.Width-4) // L/R border + padding
		m.input.SetWidth(max(1, inner-lipgloss.Width(m.input.Prompt)))
		m.viewport.SetWidth(msg.Width)
		m.ready = true
		m.layoutChrome()
		return m, nil

	case tickMsg:
		if !m.busy {
			return m, nil
		}
		m.spin++
		return m, waitTick()

	case tea.MouseWheelMsg:
		if m.scrollMouse(msg) {
			return m, nil
		}

	case streamStartMsg:
		if msg.id != m.streamID {
			return m, waitStream(msg.id, msg.ch)
		}
		if msg.err != nil {
			m.busy = false
			m.stopStream()
			m.append("error", msg.err.Error())
			return m, nil
		}
		m.streamCh = msg.ch
		return m, waitStream(msg.id, msg.ch)

	case streamDeltaMsg:
		if msg.id != m.streamID {
			return m, waitStream(msg.id, msg.ch)
		}
		switch msg.ev.Kind {
		case core.EventText:
			if m.streamCompleted {
				return m, m.failStream(msg.id, msg.ch, "provider stream: text after completion")
			}
			m.streamRaw += msg.ev.Text
			m.paintAssistant(m.streamRaw, false)
		case core.EventComplete:
			if m.streamCompleted {
				return m, m.failStream(msg.id, msg.ch, "provider stream: duplicate completion")
			}
			m.streamCompleted = true
			m.lastUsage = msg.ev.Usage
			m.sessionUsage.Add(msg.ev.Usage)
		case core.EventError:
			err := msg.ev.Err
			if err == nil {
				err = fmt.Errorf("stream error")
			}
			return m, m.failStream(msg.id, msg.ch, err.Error())
		}
		return m, waitStream(msg.id, msg.ch)

	case streamDoneMsg:
		if msg.id != m.streamID || !m.busy {
			return m, nil
		}
		m.busy = false
		if m.streamRaw != "" {
			m.paintAssistant(m.streamRaw, true)
		}
		if !m.streamCompleted {
			m.stopStream()
			m.append("error", "provider stream: closed without completion")
			return m, nil
		}
		m.history = append(m.history, core.Message{Role: core.RoleAssistant, Content: m.streamRaw})
		m.stopStream()
		return m, nil

	case tea.KeyPressMsg:
		switch msg.String() {
		case "ctrl+c":
			m.stopStream()
			return m, tea.Quit
		case "esc":
			if m.busy {
				if m.streamRaw != "" {
					m.history = append(m.history, core.Message{Role: core.RoleAssistant, Content: m.streamRaw})
					m.paintAssistant(m.streamRaw, true)
				}
				oldID, oldCh := m.abandonStream()
				m.busy = false
				m.append("system", "interrupted")
				return m, waitStream(oldID, oldCh)
			}
		case "pgup", "ctrl+u":
			if m.scrollHistory(-1, true) {
				return m, nil
			}
		case "pgdown", "ctrl+d":
			if m.scrollHistory(1, true) {
				return m, nil
			}
		case "tab":
			m.acceptSlash()
			return m, nil
		case "shift+tab", "up", "ctrl+p":
			if len(m.slashHits) > 0 {
				m.slashSel--
				if m.slashSel < 0 {
					m.slashSel = len(m.slashHits) - 1
				}
				return m, nil
			}
			if msg.String() == "up" && m.scrollHistory(-1, false) {
				return m, nil
			}
		case "down", "ctrl+n":
			if len(m.slashHits) > 0 {
				m.slashSel = (m.slashSel + 1) % len(m.slashHits)
				return m, nil
			}
			if msg.String() == "down" && m.scrollHistory(1, false) {
				return m, nil
			}
		case "enter":
			if m.busy {
				return m, nil
			}
			line := strings.TrimSpace(m.input.Value())
			if name, _ := splitSlash(line); name != "" && len(m.slashHits) > 0 {
				line = m.slashHits[m.slashSel].token()
			}
			m.input.Reset()
			m.refreshSlash()
			m.layoutChrome()
			if line == "" || line == "/" {
				return m, nil
			}
			return m.handleLine(line)
		}
	}

	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	m.refreshSlash()
	m.layoutChrome()
	return m, cmd
}

func (m model) handleLine(line string) (tea.Model, tea.Cmd) {
	if strings.HasPrefix(line, "/") {
		name, _ := splitSlash(line)
		if name != "config" {
			m.append("", line)
		}
		return m.handleSlash(line)
	}
	m.append("", line)
	if m.provider == nil {
		err := m.providerErr
		if err == nil {
			err = fmt.Errorf("no provider")
		}
		m.append("error", err.Error())
		return m, nil
	}
	m.history = append(m.history, core.Message{Role: core.RoleUser, Content: line})
	messages := append([]core.Message(nil), m.history...)
	m.streamID++
	id := m.streamID
	m.busy = true
	m.follow = true
	m.workStarted = time.Now()
	m.spin = 0
	m.streamRaw = ""
	m.streamLine = -1
	m.streamCompleted = false
	p := m.provider
	modelID := m.cfg.Model
	reasoning := m.cfg.Reasoning
	ctx, cancel := context.WithCancel(context.Background())
	m.streamCancel = cancel
	return m, tea.Batch(func() tea.Msg {
		ch, err := p.Stream(ctx, core.StreamRequest{
			Messages:  messages,
			Model:     modelID,
			Reasoning: reasoning,
		})
		return streamStartMsg{id: id, ch: ch, err: err}
	}, waitTick())
}

func (m model) handleSlash(line string) (tea.Model, tea.Cmd) {
	name, _ := splitSlash(line)
	switch name {
	case "quit", "exit", "q":
		return m, tea.Quit
	case "help", "h":
		m.append("system", helpText)
		return m, nil
	case "config":
		res, err := config.Load(m.loadArgs)
		if err != nil {
			m.append("error", err.Error())
			return m, nil
		}
		m.cfg = res.Config
		m.userPath = res.UserPath
		m.projectPath = res.ProjectPath
		if m.newProvider != nil {
			p, perr := m.newProvider(m.cfg)
			m.provider = p
			m.providerErr = perr
			if perr != nil {
				m.append("error", perr.Error())
			}
		}
		m.append("", config.Dump(res.UserPath, res.ProjectPath))
		return m, nil
	default:
		m.append("error", fmt.Sprintf("unknown command /%s  —  /help", name))
		return m, nil
	}
}

func (m *model) paintAssistant(text string, done bool) {
	display := text
	if done {
		display = renderMarkdown(text, m.width)
	}
	line := display
	if m.streamLine >= 0 && m.streamLine < len(m.lines) {
		m.lines[m.streamLine] = line
	} else {
		m.lines = append(m.lines, line)
		m.streamLine = len(m.lines) - 1
	}
	m.layoutChrome()
}

func (m *model) stopStream() {
	if m.streamCancel != nil {
		m.streamCancel()
		m.streamCancel = nil
	}
	m.streamCh = nil
	m.streamLine = -1
	m.streamCompleted = false
}

func (m *model) abandonStream() (uint64, <-chan core.StreamEvent) {
	id := m.streamID
	ch := m.streamCh
	m.streamID++
	m.stopStream()
	return id, ch
}

func (m *model) failStream(id uint64, ch <-chan core.StreamEvent, message string) tea.Cmd {
	m.busy = false
	m.abandonStream()
	m.append("error", message)
	return waitStream(id, ch)
}

func waitStream(id uint64, ch <-chan core.StreamEvent) tea.Cmd {
	if ch == nil {
		return func() tea.Msg { return streamDoneMsg{id: id} }
	}
	return func() tea.Msg {
		ev, ok := <-ch
		if !ok {
			return streamDoneMsg{id: id}
		}
		return streamDeltaMsg{id: id, ev: ev, ch: ch}
	}
}

func (m *model) append(who, text string) {
	if who == "" {
		m.lines = append(m.lines, text)
	} else {
		m.lines = append(m.lines, who+" "+text)
	}
	m.syncViewport()
}

func (m *model) syncViewport() {
	if !m.ready {
		return
	}
	m.viewport.SetContent(m.historyBody())
	if m.follow {
		m.viewport.GotoBottom()
	}
}

func waitTick() tea.Cmd {
	return tea.Tick(80*time.Millisecond, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (m model) overflowing() bool {
	if !m.ready {
		return false
	}
	body := m.historyBody()
	if body == "" {
		return false
	}
	return lipgloss.Height(body) > max(1, m.height-lipgloss.Height(m.inputChrome()))
}

func (m *model) scrollHistory(dir int, page bool) bool {
	if !m.overflowing() {
		return false
	}
	if page {
		if dir < 0 {
			m.viewport.PageUp()
		} else {
			m.viewport.PageDown()
		}
	} else if dir < 0 {
		m.viewport.ScrollUp(1)
	} else {
		m.viewport.ScrollDown(1)
	}
	m.follow = m.viewport.AtBottom()
	return true
}

func (m *model) scrollMouse(msg tea.MouseWheelMsg) bool {
	if !m.overflowing() {
		return false
	}
	switch msg.Button {
	case tea.MouseWheelUp:
		m.viewport.ScrollUp(3)
	case tea.MouseWheelDown:
		m.viewport.ScrollDown(3)
	default:
		return false
	}
	m.follow = m.viewport.AtBottom()
	return true
}

func (m model) historyBody() string {
	if len(m.lines) == 0 {
		return ""
	}
	body := strings.Join(m.lines, "\n\n")
	if w := m.viewport.Width(); w > 0 {
		return lipgloss.NewStyle().Width(w).Render(body)
	}
	return body
}

// threadAbove is the conversation above the prompt. Short threads sit at the
// top so the input follows the last message; long threads scroll in a viewport
// and the input docks to the bottom of the screen.
func (m model) threadAbove() (string, int) {
	body := m.historyBody()
	if body == "" {
		return "", 0
	}
	h := lipgloss.Height(body)
	avail := max(1, m.height-lipgloss.Height(m.inputChrome()))
	if h > avail {
		return m.viewport.View(), m.viewport.Height()
	}
	return body, h
}

func (m *model) refreshSlash() {
	hits := filterSlash(m.input.Value())
	if !slashSame(hits, m.slashHits) {
		m.slashSel = 0
	}
	m.slashHits = hits
	if m.slashSel >= len(m.slashHits) {
		m.slashSel = max(0, len(m.slashHits)-1)
	}
}

func (m *model) acceptSlash() {
	if len(m.slashHits) == 0 {
		return
	}
	m.input.SetValue(m.slashHits[m.slashSel].token())
	m.input.CursorEnd()
	m.refreshSlash()
	m.layoutChrome()
}

func (m *model) layoutChrome() {
	if !m.ready {
		return
	}
	m.viewport.SetHeight(max(1, m.height-lipgloss.Height(m.inputChrome())))
	m.syncViewport()
}

func slashSame(a, b []slashCmd) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].name != b[i].name {
			return false
		}
	}
	return true
}

func (m model) inputChrome() string {
	w := m.width
	if w <= 0 {
		w = 40
	}
	box := inputBoxStyle.Width(max(1, w-2)).Render(m.input.View())
	menu := m.slashMenuView()
	status := statusStyle.MaxWidth(w).Render(formatStatusLine(statusState{
		Model:     m.cfg.Model,
		Reasoning: m.cfg.Reasoning,
		Last:      m.lastUsage,
		Session:   m.sessionUsage,
	}))
	parts := []string{box}
	if menu != "" {
		parts = append(parts, menu)
	}
	if work := m.workingLine(); work != "" {
		parts = append(parts, workingStyle.Render(work))
	}
	parts = append(parts, status)
	return lipgloss.JoinVertical(lipgloss.Left, parts...)
}

func (m model) workingLine() string {
	if !m.busy {
		return ""
	}
	label := "Working"
	if m.streamRaw != "" {
		label = "Streaming"
	} else if m.cfg.Reasoning != "" && m.cfg.Reasoning != "none" {
		label = "Thinking"
	}
	frame := spinFrames[m.spin%len(spinFrames)]
	return fmt.Sprintf("%s %s · %s  esc to interrupt", frame, label, formatElapsed(time.Since(m.workStarted)))
}

func formatElapsed(d time.Duration) string {
	sec := int(d.Seconds())
	if sec < 60 {
		return fmt.Sprintf("%ds", sec)
	}
	return fmt.Sprintf("%dm %02ds", sec/60, sec%60)
}

func (m model) slashMenuView() string {
	if len(m.slashHits) == 0 {
		return ""
	}
	rows := make([]string, len(m.slashHits))
	for i, c := range m.slashHits {
		line := c.token()
		if i == m.slashSel {
			rows[i] = slashSelStyle.Render(line)
		} else {
			rows[i] = slashItemStyle.Render(line)
		}
	}
	return slashMenuStyle.Render(strings.Join(rows, "\n"))
}

func (m model) View() tea.View {
	chrome := m.inputChrome()
	above, aboveH := "", 0
	if m.ready {
		above, aboveH = m.threadAbove()
	}
	str := chrome
	if above != "" {
		str = lipgloss.JoinVertical(lipgloss.Left, above, chrome)
	}
	v := tea.NewView(str)
	if c := m.input.Cursor(); c != nil {
		// 1-cell border + 1-cell left padding around the textinput.
		c.Position.X += 2
		c.Position.Y += aboveH + 1
		v.Cursor = c
	}
	v.AltScreen = true
	v.MouseMode = tea.MouseModeCellMotion
	return v
}

func splitSlash(line string) (name, args string) {
	line = strings.TrimPrefix(line, "/")
	name, rest, ok := strings.Cut(line, " ")
	if !ok {
		return strings.ToLower(line), ""
	}
	return strings.ToLower(name), strings.TrimSpace(rest)
}

func renderMarkdown(text string, width int) string {
	if width < 20 {
		width = 80
	}
	r, err := glamour.NewTermRenderer(
		glamour.WithStandardStyle("dark"),
		glamour.WithWordWrap(width),
	)
	if err != nil {
		return text
	}
	out, err := r.Render(text)
	if err != nil {
		return text
	}
	return strings.TrimRight(out, "\n")
}
