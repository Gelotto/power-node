package client

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Gelotto/power-node/internal/models"
)

// TestNewAPIClient verifies the constructor creates a properly configured client
func TestNewAPIClient(t *testing.T) {
	client := NewAPIClient("https://api.example.com", "wk_test_key")

	if client.baseURL != "https://api.example.com" {
		t.Errorf("baseURL = %q, want %q", client.baseURL, "https://api.example.com")
	}
	if client.apiKey != "wk_test_key" {
		t.Errorf("apiKey = %q, want %q", client.apiKey, "wk_test_key")
	}
	if client.httpClient == nil {
		t.Error("httpClient should not be nil")
	}
	if client.httpClient.Timeout != 5*time.Minute {
		t.Errorf("httpClient.Timeout = %v, want %v", client.httpClient.Timeout, 5*time.Minute)
	}
}

// TestSetAPIKey verifies the API key can be updated
func TestSetAPIKey(t *testing.T) {
	client := NewAPIClient("https://api.example.com", "wk_old_key")
	client.SetAPIKey("wk_new_key")

	if client.apiKey != "wk_new_key" {
		t.Errorf("apiKey = %q, want %q", client.apiKey, "wk_new_key")
	}
}

// TestHealthCheck_Success tests successful health check
func TestHealthCheck_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodGet {
			t.Errorf("unexpected method: %s", r.Method)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	err := client.HealthCheck(context.Background())

	if err != nil {
		t.Errorf("HealthCheck() error = %v, want nil", err)
	}
}

// TestHealthCheck_ServerError tests health check with server error
func TestHealthCheck_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	err := client.HealthCheck(context.Background())

	if err == nil {
		t.Error("HealthCheck() error = nil, want error")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error should contain status code, got: %v", err)
	}
}

// TestHealthCheck_ConnectionRefused tests health check when server is unavailable
func TestHealthCheck_ConnectionRefused(t *testing.T) {
	client := NewAPIClient("http://localhost:99999", "wk_test_key")
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	err := client.HealthCheck(ctx)

	if err == nil {
		t.Error("HealthCheck() error = nil, want connection error")
	}
}

// TestHeartbeat_Success tests successful heartbeat with all fields
func TestHeartbeat_Success(t *testing.T) {
	var receivedBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/workers/heartbeat" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}
		if r.Header.Get("Authorization") != "Bearer wk_test_key" {
			t.Errorf("unexpected auth header: %s", r.Header.Get("Authorization"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("unexpected content type: %s", r.Header.Get("Content-Type"))
		}

		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &receivedBody)

		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")

	vram := 24
	maxRes := 1536
	maxSteps := 8
	gpuUtil := 50
	gpuMem := 8000
	gpuTemp := 65
	supportsVideo := true
	videoMaxDur := 5
	videoMaxFPS := 24
	videoMaxW := 832
	videoMaxH := 480
	supportsFaceSwap := true
	faceSwapMaxFrames := 100
	faceSwapMaxW := 2048
	faceSwapMaxH := 2048
	faceSwapEnhance := true

	data := &HeartbeatData{
		Hostname:          "test-host",
		GPUInfo:           "NVIDIA RTX 4090",
		VRAM:              &vram,
		ComputeCap:        "8.9",
		ServiceMode:       "gguf",
		MaxResolution:     &maxRes,
		MaxSteps:          &maxSteps,
		GPUUtilization:    &gpuUtil,
		GPUMemoryUsed:     &gpuMem,
		GPUTemperature:    &gpuTemp,
		SupportsVideo:     &supportsVideo,
		VideoMaxDuration:  &videoMaxDur,
		VideoMaxFPS:       &videoMaxFPS,
		VideoMaxWidth:     &videoMaxW,
		VideoMaxHeight:    &videoMaxH,
		SupportsFaceSwap:  &supportsFaceSwap,
		FaceSwapMaxFrames: &faceSwapMaxFrames,
		FaceSwapMaxWidth:  &faceSwapMaxW,
		FaceSwapMaxHeight: &faceSwapMaxH,
		FaceSwapEnhance:   &faceSwapEnhance,
	}

	err := client.Heartbeat(context.Background(), "worker-123", "online", data)

	if err != nil {
		t.Errorf("Heartbeat() error = %v, want nil", err)
	}

	// Verify request body
	if receivedBody["worker_id"] != "worker-123" {
		t.Errorf("worker_id = %v, want worker-123", receivedBody["worker_id"])
	}
	if receivedBody["status"] != "online" {
		t.Errorf("status = %v, want online", receivedBody["status"])
	}
	if receivedBody["hostname"] != "test-host" {
		t.Errorf("hostname = %v, want test-host", receivedBody["hostname"])
	}
	if receivedBody["gpu_info"] != "NVIDIA RTX 4090" {
		t.Errorf("gpu_info = %v, want NVIDIA RTX 4090", receivedBody["gpu_info"])
	}
	if receivedBody["vram"] != float64(24) {
		t.Errorf("vram = %v, want 24", receivedBody["vram"])
	}
	if receivedBody["supports_video"] != true {
		t.Errorf("supports_video = %v, want true", receivedBody["supports_video"])
	}
	if receivedBody["supports_faceswap"] != true {
		t.Errorf("supports_faceswap = %v, want true", receivedBody["supports_faceswap"])
	}
}

// TestHeartbeat_MinimalData tests heartbeat with only required fields
func TestHeartbeat_MinimalData(t *testing.T) {
	var receivedBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &receivedBody)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	err := client.Heartbeat(context.Background(), "worker-123", "online", nil)

	if err != nil {
		t.Errorf("Heartbeat() error = %v, want nil", err)
	}

	// Only worker_id and status should be present
	if receivedBody["worker_id"] != "worker-123" {
		t.Errorf("worker_id = %v, want worker-123", receivedBody["worker_id"])
	}
	if receivedBody["status"] != "online" {
		t.Errorf("status = %v, want online", receivedBody["status"])
	}
	// Optional fields should not be present
	if _, ok := receivedBody["hostname"]; ok {
		t.Error("hostname should not be present when empty")
	}
}

// TestHeartbeat_ServerError tests heartbeat with server error response
func TestHeartbeat_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error": "internal server error"}`))
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	err := client.Heartbeat(context.Background(), "worker-123", "online", nil)

	if err == nil {
		t.Error("Heartbeat() error = nil, want error")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error should contain status code, got: %v", err)
	}
}

// TestClaimJob_Success tests successful job claiming
func TestClaimJob_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/workers/claim" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":     "job-123",
			"type":   "image",
			"status": "claimed",
			"prompt": "a beautiful sunset",
			"width":  1024,
			"height": 1024,
			"steps":  8,
		})
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	job, err := client.ClaimJob(context.Background(), "worker-123")

	if err != nil {
		t.Errorf("ClaimJob() error = %v, want nil", err)
	}
	if job == nil {
		t.Fatal("ClaimJob() job = nil, want job")
	}
	if job.ID != "job-123" {
		t.Errorf("job.ID = %q, want %q", job.ID, "job-123")
	}
	if job.Type != models.JobTypeImage {
		t.Errorf("job.Type = %q, want %q", job.Type, models.JobTypeImage)
	}
	if job.Prompt != "a beautiful sunset" {
		t.Errorf("job.Prompt = %q, want %q", job.Prompt, "a beautiful sunset")
	}
}

// TestClaimJob_NoContent tests claiming when queue is empty (204 response)
func TestClaimJob_NoContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	job, err := client.ClaimJob(context.Background(), "worker-123")

	if err != nil {
		t.Errorf("ClaimJob() error = %v, want nil", err)
	}
	if job != nil {
		t.Errorf("ClaimJob() job = %v, want nil (empty queue)", job)
	}
}

// TestClaimJob_VideoJob tests claiming a video job
func TestClaimJob_VideoJob(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		duration := 5
		fps := 24
		totalFrames := 120

		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":               "video-job-123",
			"type":             "video",
			"status":           "claimed",
			"prompt":           "a flying bird",
			"width":            832,
			"height":           480,
			"steps":            25,
			"duration_seconds": duration,
			"fps":              fps,
			"total_frames":     totalFrames,
		})
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	job, err := client.ClaimJob(context.Background(), "worker-123")

	if err != nil {
		t.Errorf("ClaimJob() error = %v, want nil", err)
	}
	if job == nil {
		t.Fatal("ClaimJob() job = nil, want video job")
	}
	if job.Type != models.JobTypeVideo {
		t.Errorf("job.Type = %q, want %q", job.Type, models.JobTypeVideo)
	}
	if job.DurationSeconds == nil || *job.DurationSeconds != 5 {
		t.Errorf("job.DurationSeconds = %v, want 5", job.DurationSeconds)
	}
}

// TestClaimJob_FaceSwapJob tests claiming a face-swap job
func TestClaimJob_FaceSwapJob(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":               "faceswap-job-123",
			"type":             "face_swap",
			"status":           "claimed",
			"prompt":           "",
			"width":            1024,
			"height":           1024,
			"steps":            0,
			"source_image_url": "https://example.com/source.jpg",
			"target_image_url": "https://example.com/target.jpg",
			"is_gif":           false,
			"swap_all_faces":   true,
			"enhance_result":   true,
		})
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	job, err := client.ClaimJob(context.Background(), "worker-123")

	if err != nil {
		t.Errorf("ClaimJob() error = %v, want nil", err)
	}
	if job == nil {
		t.Fatal("ClaimJob() job = nil, want face-swap job")
	}
	if job.Type != models.JobTypeFaceSwap {
		t.Errorf("job.Type = %q, want %q", job.Type, models.JobTypeFaceSwap)
	}
	if job.SourceImageURL == nil || *job.SourceImageURL != "https://example.com/source.jpg" {
		t.Errorf("job.SourceImageURL = %v, want https://example.com/source.jpg", job.SourceImageURL)
	}
}

// TestClaimJob_ServerError tests claiming with server error
func TestClaimJob_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error": "database connection failed"}`))
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	job, err := client.ClaimJob(context.Background(), "worker-123")

	if err == nil {
		t.Error("ClaimJob() error = nil, want error")
	}
	if job != nil {
		t.Errorf("ClaimJob() job = %v, want nil", job)
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error should contain status code, got: %v", err)
	}
}

// TestClaimJob_Unauthorized tests claiming with invalid API key
func TestClaimJob_Unauthorized(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error": "invalid API key"}`))
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_invalid_key")
	job, err := client.ClaimJob(context.Background(), "worker-123")

	if err == nil {
		t.Error("ClaimJob() error = nil, want error")
	}
	if job != nil {
		t.Errorf("ClaimJob() job = %v, want nil", job)
	}
	if !strings.Contains(err.Error(), "401") {
		t.Errorf("error should contain status code, got: %v", err)
	}
}

// TestClaimJob_MalformedJSON tests claiming with malformed JSON response
func TestClaimJob_MalformedJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"id": "job-123", "type": invalid_json`))
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	job, err := client.ClaimJob(context.Background(), "worker-123")

	if err == nil {
		t.Error("ClaimJob() error = nil, want JSON decode error")
	}
	if job != nil {
		t.Errorf("ClaimJob() job = %v, want nil", job)
	}
}

// TestStartProcessing_Success tests successful processing notification
func TestStartProcessing_Success(t *testing.T) {
	var receivedBody map[string]string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/workers/processing" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &receivedBody)

		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	err := client.StartProcessing(context.Background(), "worker-123", "job-456")

	if err != nil {
		t.Errorf("StartProcessing() error = %v, want nil", err)
	}
	if receivedBody["worker_id"] != "worker-123" {
		t.Errorf("worker_id = %v, want worker-123", receivedBody["worker_id"])
	}
	if receivedBody["job_id"] != "job-456" {
		t.Errorf("job_id = %v, want job-456", receivedBody["job_id"])
	}
}

// TestReportProgress_Success tests successful progress reporting
func TestReportProgress_Success(t *testing.T) {
	var receivedBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/workers/progress" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &receivedBody)

		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	err := client.ReportProgress(context.Background(), "worker-123", "job-456", 60)

	if err != nil {
		t.Errorf("ReportProgress() error = %v, want nil", err)
	}
	if receivedBody["worker_id"] != "worker-123" {
		t.Errorf("worker_id = %v, want worker-123", receivedBody["worker_id"])
	}
	if receivedBody["job_id"] != "job-456" {
		t.Errorf("job_id = %v, want job-456", receivedBody["job_id"])
	}
	if receivedBody["frames_completed"] != float64(60) {
		t.Errorf("frames_completed = %v, want 60", receivedBody["frames_completed"])
	}
}

// TestCompleteJob_Image_Success tests successful image job completion
func TestCompleteJob_Image_Success(t *testing.T) {
	var receivedBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/workers/complete" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &receivedBody)

		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	imageData := "base64encodedimagedata..."
	err := client.CompleteJob(context.Background(), "worker-123", "job-456", imageData, false, 5000)

	if err != nil {
		t.Errorf("CompleteJob() error = %v, want nil", err)
	}
	if receivedBody["worker_id"] != "worker-123" {
		t.Errorf("worker_id = %v, want worker-123", receivedBody["worker_id"])
	}
	if receivedBody["job_id"] != "job-456" {
		t.Errorf("job_id = %v, want job-456", receivedBody["job_id"])
	}
	if receivedBody["image_data"] != imageData {
		t.Errorf("image_data = %v, want %v", receivedBody["image_data"], imageData)
	}
	if _, hasVideo := receivedBody["video_data"]; hasVideo {
		t.Error("video_data should not be present for image job")
	}
	if receivedBody["generation_ms"] != float64(5000) {
		t.Errorf("generation_ms = %v, want 5000", receivedBody["generation_ms"])
	}
}

// TestCompleteJob_Video_Success tests successful video job completion
func TestCompleteJob_Video_Success(t *testing.T) {
	var receivedBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &receivedBody)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	videoData := "base64encodedvideodata..."
	err := client.CompleteJob(context.Background(), "worker-123", "job-456", videoData, true, 120000)

	if err != nil {
		t.Errorf("CompleteJob() error = %v, want nil", err)
	}
	if receivedBody["video_data"] != videoData {
		t.Errorf("video_data = %v, want %v", receivedBody["video_data"], videoData)
	}
	if _, hasImage := receivedBody["image_data"]; hasImage {
		t.Error("image_data should not be present for video job")
	}
	if receivedBody["generation_ms"] != float64(120000) {
		t.Errorf("generation_ms = %v, want 120000", receivedBody["generation_ms"])
	}
}

// TestCompleteFaceSwapJob_Success tests successful face-swap job completion
func TestCompleteFaceSwapJob_Success(t *testing.T) {
	var receivedBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &receivedBody)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	imageData := "base64encodedfaceswapresult..."
	err := client.CompleteFaceSwapJob(context.Background(), "worker-123", "job-456", imageData, "jpeg", 3000)

	if err != nil {
		t.Errorf("CompleteFaceSwapJob() error = %v, want nil", err)
	}
	if receivedBody["worker_id"] != "worker-123" {
		t.Errorf("worker_id = %v, want worker-123", receivedBody["worker_id"])
	}
	if receivedBody["job_id"] != "job-456" {
		t.Errorf("job_id = %v, want job-456", receivedBody["job_id"])
	}
	if receivedBody["image_data"] != imageData {
		t.Errorf("image_data = %v, want %v", receivedBody["image_data"], imageData)
	}
	if receivedBody["format"] != "jpeg" {
		t.Errorf("format = %v, want jpeg", receivedBody["format"])
	}
	if receivedBody["generation_ms"] != float64(3000) {
		t.Errorf("generation_ms = %v, want 3000", receivedBody["generation_ms"])
	}
}

// TestCompleteFaceSwapJob_GIF_Success tests successful GIF face-swap completion
func TestCompleteFaceSwapJob_GIF_Success(t *testing.T) {
	var receivedBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &receivedBody)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	err := client.CompleteFaceSwapJob(context.Background(), "worker-123", "job-456", "base64gif...", "gif", 15000)

	if err != nil {
		t.Errorf("CompleteFaceSwapJob() error = %v, want nil", err)
	}
	if receivedBody["format"] != "gif" {
		t.Errorf("format = %v, want gif", receivedBody["format"])
	}
}

// TestFailJob_Success tests successful job failure reporting
func TestFailJob_Success(t *testing.T) {
	var receivedBody map[string]string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/workers/fail" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &receivedBody)

		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")
	err := client.FailJob(context.Background(), "worker-123", "job-456", "CUDA out of memory")

	if err != nil {
		t.Errorf("FailJob() error = %v, want nil", err)
	}
	if receivedBody["worker_id"] != "worker-123" {
		t.Errorf("worker_id = %v, want worker-123", receivedBody["worker_id"])
	}
	if receivedBody["job_id"] != "job-456" {
		t.Errorf("job_id = %v, want job-456", receivedBody["job_id"])
	}
	if receivedBody["error"] != "CUDA out of memory" {
		t.Errorf("error = %v, want 'CUDA out of memory'", receivedBody["error"])
	}
}

// TestAuthorizationHeader tests that auth header is set correctly
func TestAuthorizationHeader(t *testing.T) {
	tests := []struct {
		name   string
		apiKey string
		want   string
	}{
		{
			name:   "with API key",
			apiKey: "wk_test_key_12345",
			want:   "Bearer wk_test_key_12345",
		},
		{
			name:   "empty API key",
			apiKey: "",
			want:   "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var receivedAuth string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedAuth = r.Header.Get("Authorization")
				w.WriteHeader(http.StatusOK)
			}))
			defer server.Close()

			client := NewAPIClient(server.URL, tt.apiKey)
			_ = client.Heartbeat(context.Background(), "worker-123", "online", nil)

			if receivedAuth != tt.want {
				t.Errorf("Authorization header = %q, want %q", receivedAuth, tt.want)
			}
		})
	}
}

// TestContextCancellation tests that context cancellation is respected
func TestContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate slow response
		time.Sleep(5 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	err := client.Heartbeat(ctx, "worker-123", "online", nil)

	if err == nil {
		t.Error("Heartbeat() error = nil, want context deadline exceeded")
	}
}

// TestLargePayload tests handling of large base64 payloads
func TestLargePayload(t *testing.T) {
	var receivedSize int

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		receivedSize = len(body)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewAPIClient(server.URL, "wk_test_key")

	// Create a 1MB payload (simulating a large image)
	largeData := strings.Repeat("A", 1024*1024)
	err := client.CompleteJob(context.Background(), "worker-123", "job-456", largeData, false, 5000)

	if err != nil {
		t.Errorf("CompleteJob() error = %v, want nil", err)
	}

	// Verify the payload was received (JSON wrapping adds some overhead)
	if receivedSize < 1024*1024 {
		t.Errorf("received payload size = %d, want at least 1MB", receivedSize)
	}
}
