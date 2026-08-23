package core

func CollectText(ch <-chan StreamEvent) (string, error) {
	response, err := CollectResponse(ch)
	return response.Text, err
}
