package worker

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/Gelotto/power-node/internal/client"
	"github.com/Gelotto/power-node/internal/executor"
	"github.com/Gelotto/power-node/internal/models"
)

// Worker represents the main worker agent
type Worker struct {
	id           string
	hostname     string
	gpuInfo      string
	capabilities *GPUCapabilities
	config       *Config
	configPath   string
	apiClient    *client.APIClient
	pythonExec   *executor.PythonExecutor
	stopChan     chan struct{}
	// GPU idle detection
	gpuMonitor *GPUMonitor
	isPaused   bool
	pauseMu    sync.RWMutex
	// Python lifecycle management (for VRAM release)
	pythonMu        sync.Mutex // Protects pythonExec and pythonRunning
	pythonRunning   bool
	processingJob   bool
	processingJobMu sync.RWMutex
	// Video capability
	supportsVideo bool
}

// NewWorker creates a new worker
func NewWorker(config *Config, configPath string) (*Worker, error) {
	apiClient := client.NewAPIClient(config.API.URL, config.API.Key)

	return &Worker{
		id:         config.Worker.ID,
		config:     config,
		configPath: configPath,
		apiClient:  apiClient,
		stopChan:   make(chan struct{}),
	}, nil
}

// Start starts the worker
func (w *Worker) Start(ctx context.Context) error {
	log.Println("Starting worker...")

	w.hostname = w.config.Worker.Hostname
	if w.hostname == "" {
		var err error
		w.hostname, err = os.Hostname()
		if err != nil {
			w.hostname = "unknown"
		}
	}

	// Detect GPU capabilities
	caps, err := DetectGPUCapabilities()
	if err != nil {
		log.Printf("Warning: Could not detect GPU capabilities: %v", err)
		// Fall back to basic detection
		w.gpuInfo = w.config.Worker.GPUInfo
		if w.gpuInfo == "" {
			w.gpuInfo = w.detectGPU()
		}
	} else {
		w.capabilities = caps
		w.gpuInfo = caps.GPUModel
		log.Printf("Detected GPU: %s", caps.String())
	}

	// Detect video capability
	w.detectVideoCapability()

	// Validate configuration before starting
	if err := w.config.Validate(); err != nil {
		return fmt.Errorf("%v\n\n  Please add your credentials to:\n  %s\n\n  Get your credentials at: https://gelotto.io/workers", err, w.configPath)
	}

	log.Printf("Using credentials - Worker ID: %s", w.id)
	keyPreview := w.config.API.Key
	if len(keyPreview) > 12 {
		keyPreview = keyPreview[:12] + "..."
	}
	log.Printf("API Key: %s", keyPreview)

	log.Println("Starting Python inference service...")

	scriptArgs := w.config.Python.ScriptArgs
	if len(scriptArgs) == 0 && w.config.Model.Path != "" {
		scriptArgs = []string{w.config.Model.Path}
	}

	w.pythonExec = executor.NewPythonExecutor(
		w.config.Python.Executable,
		w.config.Python.ScriptPath,
		scriptArgs,
		w.config.Python.Env,
	)

	if err := w.pythonExec.Start(
		w.config.Python.Executable,
		w.config.Python.ScriptPath,
		scriptArgs,
		w.config.Python.Env,
	); err != nil {
		return fmt.Errorf("failed to start Python service: %w", err)
	}

	log.Println("Waiting for Python service to initialize...")
	time.Sleep(10 * time.Second)
	w.pythonRunning = true

	log.Println("Worker fully initialized!")

	// Start GPU idle detection if enabled
	if w.config.Worker.IdleDetection.Enabled {
		monitor, err := NewGPUMonitor(w.config.Worker.IdleDetection)
		if err != nil {
			log.Printf("Warning: GPU idle detection disabled: %v", err)
		} else {
			w.gpuMonitor = monitor
			w.gpuMonitor.Start()
			log.Println("GPU idle detection enabled")
		}
	}

	go w.heartbeatLoop(ctx)

	return w.jobLoop(ctx)
}

// jobLoop is the main loop that processes jobs
func (w *Worker) jobLoop(ctx context.Context) error {
	ticker := time.NewTicker(w.config.Worker.PollInterval)
	defer ticker.Stop()

	log.Printf("Starting job loop (polling every %v)...", w.config.Worker.PollInterval)

	for {
		select {
		case <-ctx.Done():
			log.Println("Context cancelled, stopping job loop...")
			return ctx.Err()

		case <-w.stopChan:
			log.Println("Stop signal received, stopping job loop...")
			return nil

		// Handle pause signal from GPU monitor
		case <-w.getPauseChan():
			w.pauseMu.Lock()
			w.isPaused = true
			w.pauseMu.Unlock()
			log.Println("Paused: GPU busy with external workload")

			// Stop Python to free VRAM - SYNCHRONOUS to prevent race with resume
			// This waits for current job to complete, then stops Python
			w.stopPythonForPause()

		// Handle resume signal from GPU monitor
		case <-w.getResumeChan():
			log.Println("GPU idle - restarting Python and reloading model...")

			// Restart Python and reload model
			if err := w.restartPython(); err != nil {
				log.Printf("Failed to restart Python: %v - staying paused", err)
				continue
			}

			w.pauseMu.Lock()
			w.isPaused = false
			w.pauseMu.Unlock()
			log.Println("Resumed: Model reloaded, ready to claim jobs")

		case <-ticker.C:
			// Skip claiming if paused
			w.pauseMu.RLock()
			paused := w.isPaused
			w.pauseMu.RUnlock()
			if paused {
				continue
			}

			// Skip claiming if Python not running (safety check)
			w.pythonMu.Lock()
			running := w.pythonRunning
			w.pythonMu.Unlock()
			if !running {
				continue
			}

			job, err := w.apiClient.ClaimJob(ctx, w.id)
			if err != nil {
				log.Printf("Error claiming job: %v", err)
				continue
			}

			if job == nil {
				continue
			}

			log.Printf("Claimed job %s: %s", job.ID, job.Prompt)
			w.processJob(ctx, job)
		}
	}
}

// getPauseChan safely gets the pause channel (returns nil if no monitor)
func (w *Worker) getPauseChan() <-chan struct{} {
	if w.gpuMonitor != nil {
		return w.gpuMonitor.PauseChan()
	}
	return nil
}

// getResumeChan safely gets the resume channel (returns nil if no monitor)
func (w *Worker) getResumeChan() <-chan struct{} {
	if w.gpuMonitor != nil {
		return w.gpuMonitor.ResumeChan()
	}
	return nil
}

// isProcessingJob checks if a job is currently being processed
func (w *Worker) isProcessingJob() bool {
	w.processingJobMu.RLock()
	defer w.processingJobMu.RUnlock()
	return w.processingJob
}

// setProcessingJob sets the job processing state
func (w *Worker) setProcessingJob(processing bool) {
	w.processingJobMu.Lock()
	w.processingJob = processing
	w.processingJobMu.Unlock()
}

// stopPythonForPause stops Python to free VRAM, waiting for current job to finish
// This is called SYNCHRONOUSLY to ensure stop completes before any resume can happen
func (w *Worker) stopPythonForPause() {
	// Wait for current job to complete before stopping Python
	for w.isProcessingJob() {
		time.Sleep(1 * time.Second)
		log.Println("Waiting for current job to complete before freeing VRAM...")
	}

	// Lock to prevent race with restartPython
	w.pythonMu.Lock()
	defer w.pythonMu.Unlock()

	if w.pythonExec != nil && w.pythonRunning {
		log.Println("Stopping Python to free VRAM...")
		if err := w.pythonExec.Stop(); err != nil {
			log.Printf("Error stopping Python: %v", err)
		}
		w.pythonExec = nil
		w.pythonRunning = false
		log.Println("VRAM freed - user can now use GPU for other applications")
	}
}

// restartPython restarts the Python inference service and waits for model to load
func (w *Worker) restartPython() error {
	w.pythonMu.Lock()
	defer w.pythonMu.Unlock()

	// Safety: ensure old executor is fully stopped (defensive)
	if w.pythonExec != nil {
		log.Println("Cleaning up old Python executor...")
		w.pythonExec.Stop()
		w.pythonExec = nil
	}

	scriptArgs := w.config.Python.ScriptArgs
	if len(scriptArgs) == 0 && w.config.Model.Path != "" {
		scriptArgs = []string{w.config.Model.Path}
	}

	w.pythonExec = executor.NewPythonExecutor(
		w.config.Python.Executable,
		w.config.Python.ScriptPath,
		scriptArgs,
		w.config.Python.Env,
	)

	if err := w.pythonExec.Start(
		w.config.Python.Executable,
		w.config.Python.ScriptPath,
		scriptArgs,
		w.config.Python.Env,
	); err != nil {
		return fmt.Errorf("failed to restart Python: %w", err)
	}

	// Wait for model to reload
	log.Println("Waiting for model to reload (10s)...")
	time.Sleep(10 * time.Second)
	w.pythonRunning = true

	return nil
}

// processJob processes a single job (routes to image or video processing)
func (w *Worker) processJob(ctx context.Context, job *models.Job) {
	// Track that we're processing a job (for pause coordination)
	w.setProcessingJob(true)
	defer w.setProcessingJob(false)

	// Route based on job type
	if job.IsVideoJob() {
		// Validate video capability before processing
		if !w.canProcessVideo() {
			log.Printf("Cannot process video job %s: worker does not support video generation", job.ID)
			if err := w.apiClient.FailJob(ctx, w.id, job.ID, "Worker does not support video generation"); err != nil {
				log.Printf("Failed to report job failure: %v", err)
			}
			return
		}
		w.processVideoJob(ctx, job)
	} else {
		w.processImageJob(ctx, job)
	}
}

// processImageJob processes an image generation job
func (w *Worker) processImageJob(ctx context.Context, job *models.Job) {
	log.Printf("Processing image job %s...", job.ID)
	startTime := time.Now()

	if err := w.apiClient.StartProcessing(ctx, w.id, job.ID); err != nil {
		log.Printf("Failed to notify processing start for job %s: %v", job.ID, err)
	}

	genReq := &models.GenerateRequest{
		Prompt: job.Prompt,
		Width:  job.Width,
		Height: job.Height,
		Steps:  job.Steps,
		Seed:   job.Seed,
	}

	result, err := w.pythonExec.Generate(ctx, genReq)
	if err != nil {
		log.Printf("Image generation failed for job %s: %v", job.ID, err)
		if err := w.apiClient.FailJob(ctx, w.id, job.ID, err.Error()); err != nil {
			log.Printf("Failed to report job failure: %v", err)
		}
		return
	}

	elapsed := time.Since(startTime)
	generationMs := int(elapsed.Milliseconds())
	log.Printf("Image generation completed in %.2f seconds", elapsed.Seconds())

	log.Printf("Uploading image result for job %s...", job.ID)
	if err := w.apiClient.CompleteJob(ctx, w.id, job.ID, result.ImageData, false, generationMs); err != nil {
		log.Printf("Failed to complete job %s: %v", job.ID, err)
		return
	}

	log.Printf("Image job %s completed successfully!", job.ID)
}

// processVideoJob processes a video generation job
func (w *Worker) processVideoJob(ctx context.Context, job *models.Job) {
	log.Printf("Processing video job %s...", job.ID)
	startTime := time.Now()

	if err := w.apiClient.StartProcessing(ctx, w.id, job.ID); err != nil {
		log.Printf("Failed to notify processing start for job %s: %v", job.ID, err)
	}

	// Calculate total frames from duration and FPS
	durationSeconds := 5 // Default
	if job.DurationSeconds != nil {
		durationSeconds = *job.DurationSeconds
	}
	fps := 24 // Default
	if job.FPS != nil {
		fps = *job.FPS
	}
	totalFrames := durationSeconds * fps

	// Set up progress callback to report progress to backend
	// We rate-limit progress reports to avoid overwhelming the API
	var lastReportedFrames int
	var progressMu sync.Mutex

	w.pythonExec.SetProgressCallback(func(msg models.ProgressMessage) {
		progressMu.Lock()
		defer progressMu.Unlock()

		// Report progress every 10+ frames or when nearly complete (>95%)
		shouldReport := msg.FramesCompleted >= lastReportedFrames+10 || msg.ProgressPercent >= 95
		if !shouldReport {
			return
		}

		lastReportedFrames = msg.FramesCompleted
		log.Printf("Video job %s progress: step %d/%d (%.1f%%), ~%d frames",
			job.ID, msg.Step, msg.TotalSteps, msg.ProgressPercent, msg.FramesCompleted)

		// Report to backend API (non-blocking goroutine, ignore errors)
		go func(frames int) {
			if err := w.apiClient.ReportProgress(ctx, w.id, job.ID, frames); err != nil {
				// Log but don't fail - progress reporting is best-effort
				log.Printf("Progress report failed (non-fatal): %v", err)
			}
		}(msg.FramesCompleted)
	})

	// Ensure callback is cleared when done (avoid memory leaks / stale callbacks)
	defer w.pythonExec.SetProgressCallback(nil)

	genReq := &models.GenerateVideoRequest{
		Prompt:          job.Prompt,
		NegativePrompt:  job.NegativePrompt,
		Width:           job.Width,
		Height:          job.Height,
		DurationSeconds: durationSeconds,
		FPS:             fps,
		TotalFrames:     totalFrames,
		Seed:            job.Seed,
	}

	result, err := w.pythonExec.GenerateVideo(ctx, genReq)
	if err != nil {
		log.Printf("Video generation failed for job %s: %v", job.ID, err)
		if err := w.apiClient.FailJob(ctx, w.id, job.ID, err.Error()); err != nil {
			log.Printf("Failed to report job failure: %v", err)
		}
		return
	}

	// Report final progress (100% complete)
	if err := w.apiClient.ReportProgress(ctx, w.id, job.ID, totalFrames); err != nil {
		log.Printf("Final progress report failed (non-fatal): %v", err)
	}

	elapsed := time.Since(startTime)
	generationMs := int(elapsed.Milliseconds())
	log.Printf("Video generation completed in %.2f seconds (%d frames)", elapsed.Seconds(), result.FramesGenerated)

	log.Printf("Uploading video result for job %s...", job.ID)
	if err := w.apiClient.CompleteJob(ctx, w.id, job.ID, result.VideoData, true, generationMs); err != nil {
		log.Printf("Failed to complete job %s: %v", job.ID, err)
		return
	}

	log.Printf("Video job %s completed successfully!", job.ID)
}

// heartbeatLoop sends periodic heartbeats to the backend
func (w *Worker) heartbeatLoop(ctx context.Context) {
	ticker := time.NewTicker(w.config.Worker.HeartbeatInterval)
	defer ticker.Stop()

	log.Printf("Starting heartbeat loop (every %v)...", w.config.Worker.HeartbeatInterval)

	for {
		select {
		case <-ctx.Done():
			return
		case <-w.stopChan:
			return
		case <-ticker.C:
			data := w.buildHeartbeatData()

			// Determine status based on pause state
			status := "online"
			w.pauseMu.RLock()
			if w.isPaused {
				status = "paused"
			}
			w.pauseMu.RUnlock()

			if err := w.apiClient.Heartbeat(ctx, w.id, status, data); err != nil {
				log.Printf("Heartbeat failed: %v", err)
			}
		}
	}
}

// buildHeartbeatData creates heartbeat data with capability information
func (w *Worker) buildHeartbeatData() *client.HeartbeatData {
	data := &client.HeartbeatData{
		Hostname: w.hostname,
		GPUInfo:  w.gpuInfo,
	}

	if w.capabilities != nil {
		data.VRAM = &w.capabilities.VRAM
		data.ComputeCap = w.capabilities.ComputeCap
		data.ServiceMode = w.capabilities.ServiceMode
		data.MaxResolution = &w.capabilities.MaxResolution
		data.MaxSteps = &w.capabilities.MaxSteps
	}

	// Add GPU metrics if monitoring is active
	if w.gpuMonitor != nil {
		metrics := w.gpuMonitor.GetMetrics()
		data.GPUUtilization = &metrics.Utilization
		data.GPUMemoryUsed = &metrics.MemoryUsed
		data.GPUTemperature = &metrics.Temperature
	}

	// Add video capability information
	data.SupportsVideo = &w.supportsVideo
	if w.supportsVideo {
		data.VideoMaxDuration = &w.config.Video.MaxDuration
		data.VideoMaxFPS = &w.config.Video.MaxFPS
		data.VideoMaxWidth = &w.config.Video.MaxWidth
		data.VideoMaxHeight = &w.config.Video.MaxHeight
	}

	return data
}

// Stop gracefully stops the worker
func (w *Worker) Stop() error {
	log.Println("Stopping worker...")

	close(w.stopChan)

	// Stop GPU monitor
	if w.gpuMonitor != nil {
		w.gpuMonitor.Stop()
	}

	// Stop Python with mutex protection
	w.pythonMu.Lock()
	if w.pythonExec != nil {
		if err := w.pythonExec.Stop(); err != nil {
			log.Printf("Error stopping Python executor: %v", err)
		}
		w.pythonExec = nil
		w.pythonRunning = false
	}
	w.pythonMu.Unlock()

	log.Println("Worker stopped")
	return nil
}

// detectGPU attempts to detect GPU information using nvidia-smi
func (w *Worker) detectGPU() string {
	out, err := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").Output()
	if err != nil {
		return "Unknown GPU"
	}
	gpu := strings.TrimSpace(string(out))
	if gpu == "" {
		return "Unknown GPU"
	}
	// Return first GPU if multiple
	if idx := strings.Index(gpu, "\n"); idx > 0 {
		gpu = gpu[:idx]
	}
	return gpu
}

// detectVideoCapability checks if this worker can generate videos
func (w *Worker) detectVideoCapability() {
	// Check if explicitly enabled/disabled in config
	if w.config.Video.Enabled != nil {
		w.supportsVideo = *w.config.Video.Enabled
		if w.supportsVideo {
			log.Printf("Video support: ENABLED (explicit config)")
		} else {
			log.Printf("Video support: DISABLED (explicit config)")
		}
		return
	}

	// Auto-detect based on:
	// 1. PyTorch mode (GGUF doesn't support video)
	// 2. Wan model path exists
	// 3. Sufficient VRAM (8GB+)

	// Check 1: Must be PyTorch mode
	if w.capabilities != nil && w.capabilities.ServiceMode == "gguf" {
		w.supportsVideo = false
		log.Printf("Video support: DISABLED (GGUF mode - video requires PyTorch)")
		return
	}

	// Check 2: Wan model path must be set and directory must exist
	modelPath := w.config.Video.ModelPath
	if modelPath == "" {
		w.supportsVideo = false
		log.Printf("Video support: DISABLED (WAN_MODEL_PATH not set)")
		return
	}

	// Verify the model directory exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		w.supportsVideo = false
		log.Printf("Video support: DISABLED (Wan model not found at %s)", modelPath)
		return
	}

	// Check 2b: Verify model_index.json exists (required for Diffusers pipeline)
	modelIndexPath := filepath.Join(modelPath, "model_index.json")
	if _, err := os.Stat(modelIndexPath); os.IsNotExist(err) {
		w.supportsVideo = false
		log.Printf("Video support: DISABLED (model_index.json not found - incomplete model at %s)", modelPath)
		log.Printf("  Tip: Download the Diffusers-compatible version: Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
		return
	}

	// Check 3: Sufficient VRAM (8GB minimum for Wan2.1)
	if w.capabilities != nil && w.capabilities.VRAM < 8 {
		w.supportsVideo = false
		log.Printf("Video support: DISABLED (insufficient VRAM: %dGB < 8GB)", w.capabilities.VRAM)
		return
	}

	// All checks passed
	w.supportsVideo = true
	log.Printf("Video support: ENABLED (Wan model found at %s)", modelPath)
}

// canProcessVideo checks if this worker can process video jobs
func (w *Worker) canProcessVideo() bool {
	return w.supportsVideo
}
