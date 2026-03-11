package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

type ApiClient struct {
	httpClient *http.Client
	headers    map[string]string
	BaseUrl    string
}

func NewApiHttpClient() *http.Client {
	transport := &http.Transport{
		MaxIdleConns:          100,              // Maximum idle connections
		MaxIdleConnsPerHost:   20,               // Maximum idle connections per host
		IdleConnTimeout:       90 * time.Second, // Idle connection timeout
		DisableCompression:    false,            // Enable compression
		DisableKeepAlives:     false,            // Enable keep-alives
		ResponseHeaderTimeout: 30 * time.Second,
	}

	return &http.Client{
		Transport: transport,
		Timeout:   5 * time.Minute,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			// Custom redirect handling
			if len(via) >= 10 {
				return errors.New("stopped after 10 redirects")
			}
			return nil
		},
	}
}

func NewApiClient(baseUrl string, headers map[string]string) *ApiClient {
	return &ApiClient{
		httpClient: NewApiHttpClient(),
		headers:    headers,
		BaseUrl:    baseUrl,
	}
}

func (c *ApiClient) SetHeaders(headers map[string]string) {
	c.headers = headers
}

func (c *ApiClient) newRequest(ctx context.Context, method string, endpoint string, body io.Reader) (*http.Request, error) {
	httpReq, err := http.NewRequestWithContext(ctx, method, fmt.Sprintf("%s/%s", c.BaseUrl, endpoint), body)
	if err != nil {
		return httpReq, err
	}
	for key, value := range c.headers {
		httpReq.Header.Set(key, value)
	}
	return httpReq, err
}

func (c *ApiClient) PostJson(ctx context.Context, endpoint string, data any) (*http.Response, error) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := c.newRequest(ctx, http.MethodPost, endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func (c *ApiClient) GetJson(ctx context.Context, endpoint string) (*http.Response, error) {

	httpReq, err := c.newRequest(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// func (c *ApiClient) Delete(ctx context.Context, endpoint string, data any) (*http.Response, error) {

// 	httpReq, err := c.newRequest(ctx, http.MethodDelete, endpoint, data)
// 	if err != nil {
// 		return nil, err
// 	}
// 	httpReq.Header.Set("Content-Type", "application/json")

// 	resp, err := c.httpClient.Do(httpReq)
// 	if err != nil {
// 		return nil, err
// 	}

// 	return resp, nil
// }
