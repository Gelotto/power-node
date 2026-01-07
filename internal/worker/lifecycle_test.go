package worker

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Gelotto/power-node/internal/client"
	"github.com/Gelotto/power-node/internal/testutil"
)

// ============================================================================
// WORKER LIFECYCLE TESTS
// Tests for worker initialization, heartbeat, and shutdown
// ============================================================================

func TestBuildHeartbeatData_BasicFields(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)
	worker.hostname = "test-hostname"
	worker.gpuInfo = "NVIDIA RTX 4090"

	data := worker.buildHeartbeatData()

	if data.Hostname != "test-hostname" {
		t.Errorf("Expected hostname 'test-hostname', got '%s'", data.Hostname)
	}
	if data.GPUInfo != "NVIDIA RTX 4090" {
		t.Errorf("Expected GPU info 'NVIDIA RTX 4090', got '%s'", data.GPUInfo)
	}
}

func TestBuildHeartbeatData_WithGPUCapabilities(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)
	worker.capabilities = &GPUCapabilities{
		GPUModel:      "NVIDIA RTX 4090",
		VRAM:          24,
		ComputeCap:    "8.9",
		ServiceMode:   "pytorch",
		MaxResolution: 2048,
		MaxSteps:      50,
	}

	data := worker.buildHeartbeatData()

	if data.VRAM == nil || *data.VRAM != 24 {
		t.Error("Expected VRAM 24")
	}
	if data.ComputeCap != "8.9" {
		t.Errorf("Expected ComputeCap '8.9', got '%s'", data.ComputeCap)
	}
	if data.ServiceMode != "pytorch" {
		t.Errorf("Expected ServiceMode 'pytorch', got '%s'", data.ServiceMode)
	}
	if data.MaxResolution == nil || *data.MaxResolution != 2048 {
		t.Error("Expected MaxResolution 2048")
	}
	if data.MaxSteps == nil || *data.MaxSteps != 50 {
		t.Error("Expected MaxSteps 50")
	}
}

func TestBuildHeartbeatData_VideoCapability(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)
	worker.supportsVideo = true
	worker.config = &Config{
		Video: VideoConfig{
			MaxDuration: 10,
			MaxFPS:      30,
			MaxWidth:    1920,
			MaxHeight:   1080,
		},
	}

	data := worker.buildHeartbeatData()

	if data.SupportsVideo == nil || !*data.SupportsVideo {
		t.Error("Expected SupportsVideo true")
	}
	if data.VideoMaxDuration == nil || *data.VideoMaxDuration != 10 {
		t.Error("Expected VideoMaxDuration 10")
	}
	if data.VideoMaxFPS == nil || *data.VideoMaxFPS != 30 {
		t.Error("Expected VideoMaxFPS 30")
	}
	if data.VideoMaxWidth == nil || *data.VideoMaxWidth != 1920 {
		t.Error("Expected VideoMaxWidth 1920")
	}
	if data.VideoMaxHeight == nil || *data.VideoMaxHeight != 1080 {
		t.Error("Expected VideoMaxHeight 1080")
	}
}

func TestBuildHeartbeatData_FaceSwapCapability(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)
	worker.supportsFaceSwap = true
	enhance := true
	worker.config = &Config{
		FaceSwap: FaceSwapConfig{
			MaxFrames:   100,
			MaxWidth:    1920,
			MaxHeight:   1080,
			Enhancement: &enhance,
		},
	}

	data := worker.buildHeartbeatData()

	if data.SupportsFaceSwap == nil || !*data.SupportsFaceSwap {
		t.Error("Expected SupportsFaceSwap true")
	}
	if data.FaceSwapMaxFrames == nil || *data.FaceSwapMaxFrames != 100 {
		t.Error("Expected FaceSwapMaxFrames 100")
	}
	if data.FaceSwapMaxWidth == nil || *data.FaceSwapMaxWidth != 1920 {
		t.Error("Expected FaceSwapMaxWidth 1920")
	}
	if data.FaceSwapMaxHeight == nil || *data.FaceSwapMaxHeight != 1080 {
		t.Error("Expected FaceSwapMaxHeight 1080")
	}
	if data.FaceSwapEnhance == nil || !*data.FaceSwapEnhance {
		t.Error("Expected FaceSwapEnhance true")
	}
}

func TestBuildHeartbeatData_DisabledCapabilities(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)
	worker.supportsVideo = false
	worker.supportsFaceSwap = false
	worker.config = &Config{} // Empty config

	data := worker.buildHeartbeatData()

	if data.SupportsVideo == nil || *data.SupportsVideo {
		t.Error("Expected SupportsVideo false")
	}
	if data.SupportsFaceSwap == nil || *data.SupportsFaceSwap {
		t.Error("Expected SupportsFaceSwap false")
	}

	// Video/FaceSwap limits should not be set when disabled
	if data.VideoMaxDuration != nil {
		t.Error("VideoMaxDuration should be nil when video disabled")
	}
	if data.FaceSwapMaxFrames != nil {
		t.Error("FaceSwapMaxFrames should be nil when faceswap disabled")
	}
}

func TestHeartbeatLoop_SendsHeartbeats(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	apiClient := client.NewAPIClient(mockServer.URL, testutil.ValidAPIKey())

	worker := &Worker{
		id:        "heartbeat-test-worker",
		hostname:  "test-host",
		gpuInfo:   "Test GPU",
		apiClient: apiClient,
		stopChan:  make(chan struct{}),
		config: &Config{
			Worker: WorkerConfig{
				HeartbeatInterval: 50 * time.Millisecond,
			},
		},
	}

	ctx, cancel := context.WithCancel(context.Background())

	// Start heartbeat loop in background
	go worker.heartbeatLoop(ctx)

	// Wait for some heartbeats
	testutil.WaitForCondition(t, func() bool {
		return mockServer.GetHeartbeatCalls() >= 2
	}, 500*time.Millisecond, "heartbeats to be sent")

	cancel()

	// Should have at least 2 heartbeats
	calls := mockServer.GetHeartbeatCalls()
	if calls < 2 {
		t.Errorf("Expected at least 2 heartbeats, got %d", calls)
	}
}

func TestHeartbeatLoop_StopsOnContext(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	apiClient := client.NewAPIClient(mockServer.URL, testutil.ValidAPIKey())

	worker := &Worker{
		id:        "stop-test-worker",
		hostname:  "test-host",
		apiClient: apiClient,
		stopChan:  make(chan struct{}),
		config: &Config{
			Worker: WorkerConfig{
				HeartbeatInterval: 10 * time.Millisecond,
			},
		},
	}

	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan struct{})
	go func() {
		worker.heartbeatLoop(ctx)
		close(done)
	}()

	// Wait for at least one heartbeat
	testutil.WaitForCondition(t, func() bool {
		return mockServer.GetHeartbeatCalls() >= 1
	}, 100*time.Millisecond, "first heartbeat")

	// Cancel context
	cancel()

	// Heartbeat loop should exit
	select {
	case <-done:
		// Good, loop exited
	case <-time.After(100 * time.Millisecond):
		t.Error("Heartbeat loop did not stop after context cancellation")
	}
}

func TestHeartbeatLoop_StopsOnStopChan(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	apiClient := client.NewAPIClient(mockServer.URL, testutil.ValidAPIKey())
	stopChan := make(chan struct{})

	worker := &Worker{
		id:        "stopchan-test-worker",
		hostname:  "test-host",
		apiClient: apiClient,
		stopChan:  stopChan,
		config: &Config{
			Worker: WorkerConfig{
				HeartbeatInterval: 10 * time.Millisecond,
			},
		},
	}

	ctx := context.Background()

	done := make(chan struct{})
	go func() {
		worker.heartbeatLoop(ctx)
		close(done)
	}()

	// Wait for at least one heartbeat
	testutil.WaitForCondition(t, func() bool {
		return mockServer.GetHeartbeatCalls() >= 1
	}, 100*time.Millisecond, "first heartbeat")

	// Close stop channel
	close(stopChan)

	// Heartbeat loop should exit
	select {
	case <-done:
		// Good, loop exited
	case <-time.After(100 * time.Millisecond):
		t.Error("Heartbeat loop did not stop after stopChan closed")
	}
}

func TestHeartbeatLoop_ReportsPausedStatus(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	apiClient := client.NewAPIClient(mockServer.URL, testutil.ValidAPIKey())

	worker := &Worker{
		id:        "paused-status-worker",
		hostname:  "test-host",
		apiClient: apiClient,
		stopChan:  make(chan struct{}),
		config: &Config{
			Worker: WorkerConfig{
				HeartbeatInterval: 30 * time.Millisecond,
			},
		},
	}

	// Set paused state
	worker.pauseMu.Lock()
	worker.isPaused = true
	worker.pauseMu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())

	go worker.heartbeatLoop(ctx)

	// Wait for heartbeat
	testutil.WaitForCondition(t, func() bool {
		return mockServer.GetHeartbeatCalls() >= 1
	}, 100*time.Millisecond, "heartbeat")

	cancel()

	// Verify at least one heartbeat was sent
	if mockServer.GetHeartbeatCalls() < 1 {
		t.Error("Expected at least 1 heartbeat")
	}

	// Note: To fully verify the "paused" status, we'd need to parse the request body
	// The current test just verifies heartbeats are sent when paused
}

func TestHeartbeatLoop_ContinuesOnError(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	// Make heartbeats fail
	mockServer.SetHeartbeatError(500, "Server error")

	apiClient := client.NewAPIClient(mockServer.URL, testutil.ValidAPIKey())

	worker := &Worker{
		id:        "error-resilient-worker",
		hostname:  "test-host",
		apiClient: apiClient,
		stopChan:  make(chan struct{}),
		config: &Config{
			Worker: WorkerConfig{
				HeartbeatInterval: 20 * time.Millisecond,
			},
		},
	}

	ctx, cancel := context.WithCancel(context.Background())

	go worker.heartbeatLoop(ctx)

	// Wait for multiple heartbeat attempts despite errors
	testutil.WaitForCondition(t, func() bool {
		return mockServer.GetHeartbeatCalls() >= 3
	}, 200*time.Millisecond, "multiple heartbeat attempts")

	cancel()

	// Should have continued trying despite errors
	if mockServer.GetHeartbeatCalls() < 3 {
		t.Errorf("Expected heartbeat loop to continue despite errors, got %d attempts", mockServer.GetHeartbeatCalls())
	}
}

func TestStop_StopsPythonExecutor(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	var stopCalled int32
	mockExecutor.StopFunc = func() error {
		atomic.AddInt32(&stopCalled, 1)
		return nil
	}

	err := worker.Stop()
	if err != nil {
		t.Errorf("Stop returned error: %v", err)
	}

	if atomic.LoadInt32(&stopCalled) != 1 {
		t.Errorf("Expected Stop to be called once, got %d", atomic.LoadInt32(&stopCalled))
	}

	// Verify pythonRunning flag is cleared
	if worker.pythonRunning {
		t.Error("Expected pythonRunning to be false after Stop")
	}

	// Verify pythonExec is nil
	if worker.pythonExec != nil {
		t.Error("Expected pythonExec to be nil after Stop")
	}
}

func TestStop_ClosesStopChan(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	// Start a goroutine that waits on stopChan
	stopped := make(chan struct{})
	go func() {
		<-worker.stopChan
		close(stopped)
	}()

	_ = worker.Stop()

	// stopChan should be closed
	select {
	case <-stopped:
		// Good, stop channel was closed
	case <-time.After(100 * time.Millisecond):
		t.Error("stopChan was not closed")
	}
}

func TestStop_HandlesNilPythonExecutor(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	worker := &Worker{
		stopChan:   make(chan struct{}),
		pythonExec: nil, // No executor
	}

	// Should not panic
	err := worker.Stop()
	if err != nil {
		t.Errorf("Stop returned error with nil executor: %v", err)
	}
}

func TestCanProcessVideo_RespectsFlag(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	worker.supportsVideo = true
	if !worker.canProcessVideo() {
		t.Error("canProcessVideo should return true when supportsVideo is true")
	}

	worker.supportsVideo = false
	if worker.canProcessVideo() {
		t.Error("canProcessVideo should return false when supportsVideo is false")
	}
}

func TestCanProcessFaceSwap_RespectsFlag(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	worker.supportsFaceSwap = true
	if !worker.canProcessFaceSwap() {
		t.Error("canProcessFaceSwap should return true when supportsFaceSwap is true")
	}

	worker.supportsFaceSwap = false
	if worker.canProcessFaceSwap() {
		t.Error("canProcessFaceSwap should return false when supportsFaceSwap is false")
	}
}

func TestIsProcessingJob_ThreadSafe(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	// Concurrent access should not race
	done := make(chan struct{})
	go func() {
		for i := 0; i < 1000; i++ {
			worker.setProcessingJob(true)
			worker.setProcessingJob(false)
		}
		close(done)
	}()

	// Read while writing
	for i := 0; i < 1000; i++ {
		_ = worker.isProcessingJob()
	}

	<-done
}

func TestNewWorker_InitializesCorrectly(t *testing.T) {
	config := &Config{
		API: APIConfig{
			URL: "http://test.example.com",
			Key: testutil.ValidAPIKey(),
		},
		Worker: WorkerConfig{
			ID: "test-worker-123",
		},
	}

	worker, err := NewWorker(config, "/path/to/config.toml")
	if err != nil {
		t.Fatalf("NewWorker returned error: %v", err)
	}

	if worker.id != "test-worker-123" {
		t.Errorf("Expected worker ID 'test-worker-123', got '%s'", worker.id)
	}

	if worker.config != config {
		t.Error("Worker config not set correctly")
	}

	if worker.configPath != "/path/to/config.toml" {
		t.Errorf("Expected configPath '/path/to/config.toml', got '%s'", worker.configPath)
	}

	if worker.apiClient == nil {
		t.Error("API client should be initialized")
	}

	if worker.stopChan == nil {
		t.Error("stopChan should be initialized")
	}
}

func TestGetPauseChan_ReturnsNilWithoutMonitor(t *testing.T) {
	worker := &Worker{
		gpuMonitor: nil,
	}

	if worker.getPauseChan() != nil {
		t.Error("getPauseChan should return nil without GPU monitor")
	}
}

func TestGetResumeChan_ReturnsNilWithoutMonitor(t *testing.T) {
	worker := &Worker{
		gpuMonitor: nil,
	}

	if worker.getResumeChan() != nil {
		t.Error("getResumeChan should return nil without GPU monitor")
	}
}
