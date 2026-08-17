package core

import "strings"

func CollectText(ch <-chan StreamEvent) (string, error) {
	var b strings.Builder
	for ev := range ch {
		switch ev.Kind {
		case EventText:
			b.WriteString(ev.Text)
		case EventError:
			return b.String(), ev.Err
		}
	}
	return b.String(), nil
}
