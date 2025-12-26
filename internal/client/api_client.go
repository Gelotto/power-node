package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/Gelotto/power-node/internal/models"
)

// APIClient handles communication with the backend API
type APIClient struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

// NewAPIClient creates a new API client
func NewAPIClient(baseURL, apiKey string) *APIClient {
	return &APIClient{
		baseURL: baseURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http.Transport{
				TLSHandshakeTimeout:   10 * time.Second,
				ResponseHeaderTimeout: 20 * time.Second,
				IdleConnTimeout:       90 * time.Second,
			},
		},
	}
}

// SetAPIKey updates the API key used for authentication
func (c *APIClient) SetAPIKey(apiKey string) {
	c.apiKey = apiKey
}

// HeartbeatData contains optional capability data to send with heartbeat
type HeartbeatData struct {
	Hostname      string
	GPUInfo       string
	VRAM          *int
	ComputeCap    string
	ServiceMode   string
	MaxResolution *int
	MaxSteps      *int
	// GPU metrics for idle detection
	GPUUtilization *int // Current GPU utilization (0-100)
	GPUMemoryUsed  *int // Current GPU memory used (MB)
	GPUTemperature *int // Current GPU temperature (Celsius)
}

// Heartbeat sends a heartbeat to the backend
func (c *APIClient) Heartbeat(ctx context.Context, workerID, status string, data *HeartbeatData) error {
	reqBody := map[string]interface{}{
		"worker_id": workerID,
		"status":    status,
	}

	if data != nil {
		if data.Hostname != "" {
			reqBody["hostname"] = data.Hostname
		}
		if data.GPUInfo != "" {
			reqBody["gpu_info"] = data.GPUInfo
		}
		if data.VRAM != nil {
			reqBody["vram"] = *data.VRAM
		}
		if data.ComputeCap != "" {
			reqBody["compute_cap"] = data.ComputeCap
		}
		if data.ServiceMode != "" {
			reqBody["service_mode"] = data.ServiceMode
		}
		if data.MaxResolution != nil {
			reqBody["max_resolution"] = *data.MaxResolution
		}
		if data.MaxSteps != nil {
			reqBody["max_steps"] = *data.MaxSteps
		}
		// GPU metrics for idle detection
		if data.GPUUtilization != nil {
			reqBody["gpu_utilization"] = *data.GPUUtilization
		}
		if data.GPUMemoryUsed != nil {
			reqBody["gpu_memory_used"] = *data.GPUMemoryUsed
		}
		if data.GPUTemperature != nil {
			reqBody["gpu_temperature"] = *data.GPUTemperature
		}
	}

	return c.post(ctx, "/api/v1/workers/heartbeat", reqBody, nil)
}

// ClaimJob attempts to claim a job from the queue
func (c *APIClient) ClaimJob(ctx context.Context, workerID string) (*models.Job, error) {
	reqBody := map[string]string{
		"worker_id": workerID,
	}

	resp, err := c.doRequest(ctx, "POST", "/api/v1/workers/claim", reqBody)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNoContent {
		return nil, nil
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("claim job failed: %s - %s", resp.Status, string(body))
	}

	var job models.Job
	if err := json.NewDecoder(resp.Body).Decode(&job); err != nil {
		return nil, fmt.Errorf("failed to decode job: %w", err)
	}

	return &job, nil
}

// StartProcessing notifies the backend that job processing has started
func (c *APIClient) StartProcessing(ctx context.Context, workerID, jobID string) error {
	reqBody := map[string]string{
		"worker_id": workerID,
		"job_id":    jobID,
	}

	return c.post(ctx, "/api/v1/workers/processing", reqBody, nil)
}

// CompleteJob marks a job as completed with the result
func (c *APIClient) CompleteJob(ctx context.Context, workerID, jobID, imageData string, generationMs int) error {
	reqBody := map[string]interface{}{
		"worker_id":     workerID,
		"job_id":        jobID,
		"image_data":    imageData,
		"generation_ms": generationMs,
	}

	return c.post(ctx, "/api/v1/workers/complete", reqBody, nil)
}

// FailJob marks a job as failed with an error message
func (c *APIClient) FailJob(ctx context.Context, workerID, jobID, errorMsg string) error {
	reqBody := map[string]string{
		"worker_id": workerID,
		"job_id":    jobID,
		"error":     errorMsg,
	}

	return c.post(ctx, "/api/v1/workers/fail", reqBody, nil)
}

func (c *APIClient) post(ctx context.Context, path string, reqBody interface{}, respBody interface{}) error {
	resp, err := c.doRequest(ctx, "POST", path, reqBody)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API request failed: %s - %s", resp.Status, string(body))
	}

	if respBody != nil {
		if err := json.NewDecoder(resp.Body).Decode(respBody); err != nil {
			return fmt.Errorf("failed to decode response: %w", err)
		}
	}

	return nil
}

// HealthCheck tests connectivity to the API
func (c *APIClient) HealthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("cannot reach API at %s: %w", c.baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("API returned status %d", resp.StatusCode)
	}

	return nil
}

func (c *APIClient) doRequest(ctx context.Context, method, path string, body interface{}) (*http.Response, error) {
	var reqBody io.Reader
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewBuffer(jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	return c.httpClient.Do(req)
}
