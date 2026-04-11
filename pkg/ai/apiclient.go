package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"
        "strings"
)

type ApiClient struct {
	httpClient *http.Client
	headers    map[string]string
	BaseUrl    string
	// LogPath if set will cause requests and responses to be appended to this file.
	LogPath string
	// endpoint if set will be prepended to all request paths
	endpoint string
}

// Package-level default log path used when creating new ApiClient instances.
// Callers can set this to apply a default logfile across all clients created
// with NewApiClient.
var (
	defaultLogMu   sync.Mutex
	defaultLogPath string
	// logFileMu serializes writes to logfile paths to avoid interleaved writes
	// when multiple requests stream to the same file concurrently.
	logFileMu sync.Mutex
)

// SetDefaultLogFile sets the package-level default logfile path. New clients
// created after calling this will inherit the value.
func SetDefaultLogFile(path string) {
	defaultLogMu.Lock()
	defer defaultLogMu.Unlock()
	defaultLogPath = path
}

// lockedWriter writes to a file path and serializes writes using logFileMu so
// multiple goroutines can safely write the stream contents without interleaving.
type lockedWriter struct {
	path string
}

func newLockedWriter(path string) io.Writer {
	return &lockedWriter{path: path}
}

func (w *lockedWriter) Write(p []byte) (int, error) {
	logFileMu.Lock()
	defer logFileMu.Unlock()
	f, err := os.OpenFile(w.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	return f.Write(p)
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
	defaultLogMu.Lock()
	dlp := defaultLogPath
	defaultLogMu.Unlock()

	return &ApiClient{
		httpClient: NewApiHttpClient(),
		headers:    headers,
		BaseUrl:    baseUrl,
		LogPath:    dlp,
	}
}

// WithLogFile sets the filepath where requests and responses will be logged.
// If set to an empty string (the default) no logging is performed.
func (c *ApiClient) WithLogFile(path string) *ApiClient {
	c.LogPath = path
	return c
}

// WithEndpoint sets the base endpoint path that will be prepended to all requests.
func (c *ApiClient) WithEndpoint(endpoint string) *ApiClient {
	c.endpoint = endpoint
	return c
}

func (c *ApiClient) SetHeaders(headers map[string]string) {
	c.headers = headers
}

func (c *ApiClient) newRequest(ctx context.Context, method string, endpoint string, body io.Reader) (*http.Request, error) {
	fullEndpoint := c.endpoint
	if endpoint != "" {
		if fullEndpoint != "" {
			fullEndpoint = fmt.Sprintf("%s/%s", fullEndpoint, endpoint)
		} else {
			fullEndpoint = endpoint
		}
	}
	httpReq, err := http.NewRequestWithContext(ctx, method, strings.TrimSuffix(c.BaseUrl, "/") + "/" + strings.TrimPrefix(fullEndpoint, "/"), body)
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

	// If logging enabled, wrap resp.Body with a TeeReader that writes bytes to the logfile
	// as the caller reads them. This preserves streaming semantics while capturing the stream.
	if c.LogPath != "" && resp != nil && resp.Body != nil {
		timestamp := time.Now().Format(time.RFC3339)
		w := newLockedWriter(c.LogPath)
		// Synchronously log request metadata and the response status/headers
		fmt.Fprintf(w, "-----\n[%s] REQUEST %s %s\nHeaders: %v\nBody: %s\n", timestamp, httpReq.Method, httpReq.URL.String(), httpReq.Header, string(jsonData))
		fmt.Fprintf(w, "[%s] RESPONSE %s Status=%d\nHeaders: %v\nSTREAM:\n", timestamp, httpReq.URL.String(), resp.StatusCode, resp.Header)
		// Create a TeeReader so that as the caller reads resp.Body, bytes are also written to the log.
		tee := io.TeeReader(resp.Body, w)
		resp.Body = io.NopCloser(tee)
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

	// Streaming-safe logging: wrap resp.Body in a TeeReader that writes to the locked writer
	// so stream bytes are logged as they are consumed by the caller.
	if c.LogPath != "" && resp != nil && resp.Body != nil {
		timestamp := time.Now().Format(time.RFC3339)
		w := newLockedWriter(c.LogPath)
		fmt.Fprintf(w, "-----\n[%s] REQUEST %s %s\nHeaders: %v\n", timestamp, httpReq.Method, httpReq.URL.String(), httpReq.Header)
		fmt.Fprintf(w, "[%s] RESPONSE %s Status=%d\nHeaders: %v\nSTREAM:\n", timestamp, httpReq.URL.String(), resp.StatusCode, resp.Header)
		tee := io.TeeReader(resp.Body, w)
		resp.Body = io.NopCloser(tee)
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
