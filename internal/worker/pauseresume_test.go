package worker

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Gelotto/power-node/internal/models"
	"github.com/Gelotto/power-node/internal/testutil"
)

// ============================================================================
// PAUSE/RESUME TESTS
// Tests for GPU idle detection pause/resume functionality
// ============================================================================

func TestPauseState_InitiallyFalse(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	worker.pauseMu.RLock()
	isPaused := worker.isPaused
	worker.pauseMu.RUnlock()

	if isPaused {
		t.Error("Worker should not be paused initially")
	}
}

func TestPauseState_CanBeSet(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	// Set paused
	worker.pauseMu.Lock()
	worker.isPaused = true
	worker.pauseMu.Unlock()

	worker.pauseMu.RLock()
	isPaused := worker.isPaused
	worker.pauseMu.RUnlock()

	if !isPaused {
		t.Error("Worker should be paused after setting isPaused to true")
	}
}

func TestPauseState_ThreadSafe(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	var wg sync.WaitGroup
	iterations := 1000

	// Concurrent writes
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			worker.pauseMu.Lock()
			worker.isPaused = true
			worker.pauseMu.Unlock()
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			worker.pauseMu.Lock()
			worker.isPaused = false
			worker.pauseMu.Unlock()
		}
	}()

	// Concurrent reads
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			worker.pauseMu.RLock()
			_ = worker.isPaused
			worker.pauseMu.RUnlock()
		}
	}()

	wg.Wait()
	// No race condition = test passes
}

func TestStopPythonForPause_WaitsForJob(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()

	var stopCalled int32
	mockExecutor.StopFunc = func() error {
		atomic.AddInt32(&stopCalled, 1)
		return nil
	}

	worker := createTestWorker(mockServer, mockExecutor)
	worker.pythonRunning = true

	// Simulate a job in progress
	worker.setProcessingJob(true)

	// Start stopPythonForPause in background
	done := make(chan struct{})
	go func() {
		worker.stopPythonForPause()
		close(done)
	}()

	// Verify it's waiting (stop not called yet)
	time.Sleep(50 * time.Millisecond)
	if atomic.LoadInt32(&stopCalled) != 0 {
		t.Error("Stop should not be called while job is processing")
	}

	// Finish the job
	worker.setProcessingJob(false)

	// Wait for stopPythonForPause to complete
	select {
	case <-done:
		// Good
	case <-time.After(2 * time.Second):
		t.Fatal("stopPythonForPause did not complete")
	}

	if atomic.LoadInt32(&stopCalled) != 1 {
		t.Errorf("Expected Stop to be called once, got %d", atomic.LoadInt32(&stopCalled))
	}

	if worker.pythonRunning {
		t.Error("pythonRunning should be false after stopPythonForPause")
	}
}

func TestStopPythonForPause_StopsImmediatelyWithNoJob(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()

	var stopCalled int32
	mockExecutor.StopFunc = func() error {
		atomic.AddInt32(&stopCalled, 1)
		return nil
	}

	worker := createTestWorker(mockServer, mockExecutor)
	worker.pythonRunning = true
	worker.setProcessingJob(false) // No job in progress

	start := time.Now()
	worker.stopPythonForPause()
	elapsed := time.Since(start)

	// Should complete quickly (not waiting)
	if elapsed > 500*time.Millisecond {
		t.Errorf("stopPythonForPause took too long: %v", elapsed)
	}

	if atomic.LoadInt32(&stopCalled) != 1 {
		t.Errorf("Expected Stop to be called once, got %d", atomic.LoadInt32(&stopCalled))
	}
}

func TestStopPythonForPause_HandlesNilExecutor(t *testing.T) {
	worker := &Worker{
		pythonExec:    nil,
		pythonRunning: false,
	}

	// Should not panic
	worker.stopPythonForPause()
}

func TestStopPythonForPause_SetsExecutorToNil(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)
	worker.pythonRunning = true

	worker.stopPythonForPause()

	if worker.pythonExec != nil {
		t.Error("pythonExec should be nil after stopPythonForPause")
	}
}

func TestRestartPython_CleansUpOldExecutor(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()

	var stopCalls int32
	mockExecutor.StopFunc = func() error {
		atomic.AddInt32(&stopCalls, 1)
		return nil
	}

	worker := createTestWorker(mockServer, mockExecutor)
	worker.pythonRunning = true
	worker.config = &Config{
		Python: PythonConfig{
			Executable: "python3",
			ScriptPath: "/path/to/script.py",
			ScriptArgs: []string{},
		},
		Model: ModelConfig{
			Path: "/path/to/model",
		},
	}

	// Note: restartPython will try to start a new executor, which will fail
	// in a test environment, but we're testing the cleanup logic
	_ = worker.restartPython()

	// Old executor should have been stopped
	if atomic.LoadInt32(&stopCalls) != 1 {
		t.Errorf("Expected old executor to be stopped, got %d stop calls", atomic.LoadInt32(&stopCalls))
	}
}

func TestPauseDuringJobProcessing_JobCompletes(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()

	// Simulate slow generation
	mockExecutor.GenerateFunc = func(ctx context.Context, req *models.GenerateRequest) (*models.GenerateResponse, error) {
		time.Sleep(100 * time.Millisecond)
		return &models.GenerateResponse{ImageData: "test", Format: "png"}, nil
	}

	worker := createTestWorker(mockServer, mockExecutor)
	worker.pythonRunning = true

	ctx := context.Background()
	job := testutil.CreateImageJob("pause-during-job", "test")

	// Start job processing
	jobDone := make(chan struct{})
	go func() {
		worker.processJob(ctx, job)
		close(jobDone)
	}()

	// Wait for job to start
	testutil.WaitForCondition(t, func() bool {
		return worker.isProcessingJob()
	}, 200*time.Millisecond, "job to start processing")

	// Try to pause - should wait for job
	pauseDone := make(chan struct{})
	go func() {
		worker.stopPythonForPause()
		close(pauseDone)
	}()

	// Job should still complete
	select {
	case <-jobDone:
		// Good, job completed
	case <-time.After(500 * time.Millisecond):
		t.Error("Job did not complete during pause attempt")
	}

	// Pause should complete after job (stopPythonForPause polls every 1s)
	select {
	case <-pauseDone:
		// Good, pause completed
	case <-time.After(2 * time.Second):
		t.Error("Pause did not complete after job finished")
	}

	// Job should have completed successfully
	if mockServer.GetCompleteCalls() != 1 {
		t.Errorf("Expected 1 CompleteJob call, got %d", mockServer.GetCompleteCalls())
	}
}

func TestPythonMutex_ProtectsExecutor(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)
	worker.pythonRunning = true

	var wg sync.WaitGroup
	iterations := 100

	// Concurrent access through stopPythonForPause (tries to get lock and modify)
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			worker.pythonMu.Lock()
			_ = worker.pythonRunning
			worker.pythonMu.Unlock()
		}
	}()

	// Concurrent reads
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			worker.pythonMu.Lock()
			_ = worker.pythonExec
			worker.pythonMu.Unlock()
		}
	}()

	wg.Wait()
	// No race = test passes
}

func TestRapidPauseResume_NoRaceCondition(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	iterations := 100
	var wg sync.WaitGroup

	// Rapid pause/resume state changes
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			worker.pauseMu.Lock()
			worker.isPaused = true
			worker.pauseMu.Unlock()
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			worker.pauseMu.Lock()
			worker.isPaused = false
			worker.pauseMu.Unlock()
		}
	}()

	// Concurrent reads to check consistency
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			worker.pauseMu.RLock()
			_ = worker.isPaused
			worker.pauseMu.RUnlock()
		}
	}()

	wg.Wait()
	// Test passes if no race detector errors
}

func TestPythonRunningFlag_ConsistentWithExecutor(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)
	worker.pythonRunning = true

	// Stop should clear both
	worker.stopPythonForPause()

	if worker.pythonRunning {
		t.Error("pythonRunning should be false after stop")
	}
	if worker.pythonExec != nil {
		t.Error("pythonExec should be nil after stop")
	}
}

func TestProcessingJobFlag_PreventsPrematureStop(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()

	var stopTime time.Time
	mockExecutor.StopFunc = func() error {
		stopTime = time.Now()
		return nil
	}

	worker := createTestWorker(mockServer, mockExecutor)
	worker.pythonRunning = true

	// Set job as processing
	jobDone := make(chan struct{})
	worker.setProcessingJob(true)

	startTime := time.Now()

	// Start stop in background
	go func() {
		// Simulate job taking 100ms
		time.Sleep(100 * time.Millisecond)
		worker.setProcessingJob(false)
		close(jobDone)
	}()

	worker.stopPythonForPause()

	// Stop should have waited for job
	waitDuration := stopTime.Sub(startTime)
	if waitDuration < 90*time.Millisecond {
		t.Errorf("Stop should have waited for job, waited only %v", waitDuration)
	}

	<-jobDone
}
