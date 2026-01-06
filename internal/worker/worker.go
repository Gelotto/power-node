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
	pythonExec   Executor // Interface for testability
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
	// Face-swap capability
	supportsFaceSwap bool
	// Multi-model support
	supportedModels []string     // All models this worker has installed
	activeModel     *string      // Currently loaded model (nil = none)
	activeModelMu   sync.RWMutex // Protects activeModel
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

	// Detect face-swap capability
	w.detectFaceSwapCapability()

	// Initialize multi-model support
	w.supportedModels = w.config.GetSupportedModelNames()
	if len(w.supportedModels) == 0 {
		// Fallback to legacy single model
		w.supportedModels = []string{w.config.Model.Name}
	}
	log.Printf("Supported models: %v", w.supportedModels)

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

	// Set initial active model (first model loaded at startup)
	if len(w.supportedModels) > 0 {
		initialModel := w.config.GetDefaultModelName()
		w.activeModelMu.Lock()
		w.activeModel = &initialModel
		w.activeModelMu.Unlock()
		log.Printf("Active model: %s", initialModel)
	}

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
	return w.restartPythonWithModel("")
}

// restartPythonWithModel restarts Python with a specific model, or default if empty
func (w *Worker) restartPythonWithModel(modelName string) error {
	w.pythonMu.Lock()
	defer w.pythonMu.Unlock()

	// Safety: ensure old executor is fully stopped (defensive)
	if w.pythonExec != nil {
		log.Println("Cleaning up old Python executor...")
		w.pythonExec.Stop()
		w.pythonExec = nil
	}

	// Determine model path
	var modelPath string
	if modelName != "" {
		modelDef := w.config.GetModelDefinition(modelName)
		if modelDef != nil {
			modelPath = modelDef.Path
		}
	}
	if modelPath == "" {
		// Fallback to legacy config
		modelPath = w.config.Model.Path
	}

	scriptArgs := w.config.Python.ScriptArgs
	if len(scriptArgs) == 0 && modelPath != "" {
		scriptArgs = []string{modelPath}
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

	// Update active model
	if modelName != "" {
		w.activeModelMu.Lock()
		w.activeModel = &modelName
		w.activeModelMu.Unlock()
	}

	return nil
}

// loadModel ensures the specified model is loaded, swapping if necessary
// Returns nil if model is already loaded or swap succeeded
func (w *Worker) loadModel(ctx context.Context, modelName string) error {
	// Check if model is already loaded
	w.activeModelMu.RLock()
	if w.activeModel != nil && *w.activeModel == modelName {
		w.activeModelMu.RUnlock()
		return nil // Already loaded, nothing to do
	}
	currentModel := ""
	if w.activeModel != nil {
		currentModel = *w.activeModel
	}
	w.activeModelMu.RUnlock()

	// Check if we support this model
	// If supportedModels is empty (legacy/test mode), allow any model
	if len(w.supportedModels) > 0 {
		supported := false
		for _, m := range w.supportedModels {
			if m == modelName {
				supported = true
				break
			}
		}
		if !supported {
			return fmt.Errorf("model %s not supported by this worker", modelName)
		}
	}

	// Special case: if no activeModel is set but Python is already running,
	// assume the correct model is loaded (legacy/test mode compatibility)
	w.pythonMu.Lock()
	pythonIsRunning := w.pythonExec != nil && w.pythonRunning
	w.pythonMu.Unlock()

	if currentModel == "" && pythonIsRunning {
		// Python is running but no active model tracked - set it now
		w.activeModelMu.Lock()
		w.activeModel = &modelName
		w.activeModelMu.Unlock()
		return nil
	}

	// Need to swap models - this involves restarting Python
	if currentModel != "" {
		log.Printf("Swapping model: %s -> %s (this may take 5-10 seconds)...", currentModel, modelName)
	} else {
		log.Printf("Loading model: %s...", modelName)
	}

	// Stop current Python (releases VRAM)
	w.pythonMu.Lock()
	if w.pythonExec != nil {
		log.Println("Unloading current model...")
		w.pythonExec.Stop()
		w.pythonExec = nil
		w.pythonRunning = false
	}
	w.pythonMu.Unlock()

	// Clear active model during transition
	w.activeModelMu.Lock()
	w.activeModel = nil
	w.activeModelMu.Unlock()

	// Restart with new model
	if err := w.restartPythonWithModel(modelName); err != nil {
		return fmt.Errorf("failed to load model %s: %w", modelName, err)
	}

	log.Printf("Model %s loaded successfully", modelName)
	return nil
}

// getActiveModel returns the currently loaded model name (thread-safe)
func (w *Worker) getActiveModel() string {
	w.activeModelMu.RLock()
	defer w.activeModelMu.RUnlock()
	if w.activeModel != nil {
		return *w.activeModel
	}
	return ""
}

// processJob processes a single job (routes to image, video, or face-swap processing)
func (w *Worker) processJob(ctx context.Context, job *models.Job) {
	// Track that we're processing a job (for pause coordination)
	w.setProcessingJob(true)
	defer w.setProcessingJob(false)

	// Route based on job type
	switch {
	case job.IsFaceSwapJob():
		// Validate face-swap capability before processing
		if !w.canProcessFaceSwap() {
			log.Printf("Cannot process face-swap job %s: worker does not support face-swap", job.ID)
			if err := w.apiClient.FailJob(ctx, w.id, job.ID, "Worker does not support face-swap"); err != nil {
				log.Printf("Failed to report job failure: %v", err)
			}
			return
		}
		w.processFaceSwapJob(ctx, job)
	case job.IsVideoJob():
		// Validate video capability before processing
		if !w.canProcessVideo() {
			log.Printf("Cannot process video job %s: worker does not support video generation", job.ID)
			if err := w.apiClient.FailJob(ctx, w.id, job.ID, "Worker does not support video generation"); err != nil {
				log.Printf("Failed to report job failure: %v", err)
			}
			return
		}
		w.processVideoJob(ctx, job)
	default:
		w.processImageJob(ctx, job)
	}
}

// processImageJob processes an image generation job
func (w *Worker) processImageJob(ctx context.Context, job *models.Job) {
	log.Printf("Processing image job %s...", job.ID)
	startTime := time.Now()

	// Determine required model (default to z-image-turbo for backwards compatibility)
	requiredModel := job.Model
	if requiredModel == "" {
		requiredModel = w.config.GetDefaultModelName()
		if requiredModel == "" {
			requiredModel = "z-image-turbo"
		}
	}

	// Multi-model support: load the required model if not already active
	if err := w.loadModel(ctx, requiredModel); err != nil {
		errMsg := fmt.Sprintf("failed to load model %s: %v", requiredModel, err)
		log.Printf("Rejecting job %s: %s", job.ID, errMsg)
		if err := w.apiClient.FailJob(ctx, w.id, job.ID, errMsg); err != nil {
			log.Printf("Failed to report model load failure for job %s: %v", job.ID, err)
		}
		return
	}

	if err := w.apiClient.StartProcessing(ctx, w.id, job.ID); err != nil {
		log.Printf("Failed to notify processing start for job %s: %v", job.ID, err)
	}

	genReq := &models.GenerateRequest{
		Prompt:         job.Prompt,
		NegativePrompt: job.NegativePrompt,
		Width:          job.Width,
		Height:         job.Height,
		Steps:          job.Steps,
		Seed:           job.Seed,
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

	// Note: Video generation uses Wan2.1 model, which is separate from image models
	// No image model check needed here - video capability is checked in processJob()

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

	// Pass video inference steps from job (set by backend based on tier)
	// Default to 25 if not specified (for backwards compatibility)
	var videoSteps *int
	if job.Steps > 0 {
		steps := job.Steps
		videoSteps = &steps
	}

	genReq := &models.GenerateVideoRequest{
		Prompt:          job.Prompt,
		NegativePrompt:  job.NegativePrompt,
		Width:           job.Width,
		Height:          job.Height,
		DurationSeconds: durationSeconds,
		FPS:             fps,
		TotalFrames:     totalFrames,
		Seed:            job.Seed,
		Steps:           videoSteps, // Pass steps to Python (Basic=20, Pro=35, Premium=50)
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

// processFaceSwapJob processes a face-swap job
func (w *Worker) processFaceSwapJob(ctx context.Context, job *models.Job) {
	log.Printf("Processing face-swap job %s...", job.ID)
	startTime := time.Now()

	// Note: Face-swap uses insightface model, which is separate from image models
	// No image model check needed here - face-swap capability is checked in processJob()

	if err := w.apiClient.StartProcessing(ctx, w.id, job.ID); err != nil {
		log.Printf("Failed to notify processing start for job %s: %v", job.ID, err)
	}

	// Validate required fields
	if job.SourceImageURL == nil || job.TargetImageURL == nil {
		log.Printf("Face-swap job %s missing source or target URL", job.ID)
		if err := w.apiClient.FailJob(ctx, w.id, job.ID, "Missing source or target image URL"); err != nil {
			log.Printf("Failed to report job failure: %v", err)
		}
		return
	}

	// Build face-swap request from job
	isGIF := false
	if job.IsGIF != nil {
		isGIF = *job.IsGIF
	}
	swapAll := true
	if job.SwapAllFaces != nil {
		swapAll = *job.SwapAllFaces
	}
	enhance := true
	if job.EnhanceResult != nil {
		enhance = *job.EnhanceResult
	}
	maxFrames := w.config.FaceSwap.MaxFrames
	if job.GifFrameCount != nil && *job.GifFrameCount > 0 {
		maxFrames = *job.GifFrameCount
	}

	req := &models.FaceSwapRequest{
		SourceImageURL: *job.SourceImageURL,
		TargetImageURL: *job.TargetImageURL,
		IsGIF:          isGIF,
		SwapAllFaces:   swapAll,
		Enhance:        enhance,
		MaxFrames:      maxFrames,
	}

	result, err := w.pythonExec.GenerateFaceSwap(ctx, req)
	if err != nil {
		log.Printf("Face-swap failed for job %s: %v", job.ID, err)
		if err := w.apiClient.FailJob(ctx, w.id, job.ID, err.Error()); err != nil {
			log.Printf("Failed to report job failure: %v", err)
		}
		return
	}

	elapsed := time.Since(startTime)
	generationMs := int(elapsed.Milliseconds())
	log.Printf("Face-swap completed in %.2f seconds", elapsed.Seconds())

	log.Printf("Uploading face-swap result for job %s...", job.ID)
	if err := w.apiClient.CompleteFaceSwapJob(ctx, w.id, job.ID, result.ImageData, result.Format, generationMs); err != nil {
		log.Printf("Failed to complete job %s: %v", job.ID, err)
		return
	}

	log.Printf("Face-swap job %s completed successfully!", job.ID)
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

	// Multi-model support: report all supported models and current active model
	if len(w.supportedModels) > 0 {
		data.SupportedModels = w.supportedModels
	}

	w.activeModelMu.RLock()
	if w.activeModel != nil {
		data.ActiveModel = w.activeModel
		// Also set legacy field for backwards compatibility
		data.ImageModel = *w.activeModel
	}
	w.activeModelMu.RUnlock()

	// Legacy single-model fallback
	if data.ImageModel == "" && w.config != nil {
		data.ImageModel = w.config.Model.Name
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

	// Add face-swap capability information
	data.SupportsFaceSwap = &w.supportsFaceSwap
	if w.supportsFaceSwap {
		data.FaceSwapMaxFrames = &w.config.FaceSwap.MaxFrames
		data.FaceSwapMaxWidth = &w.config.FaceSwap.MaxWidth
		data.FaceSwapMaxHeight = &w.config.FaceSwap.MaxHeight
		data.FaceSwapEnhance = w.config.FaceSwap.Enhancement
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

// detectFaceSwapCapability checks if this worker can perform face-swaps
func (w *Worker) detectFaceSwapCapability() {
	// Check if explicitly enabled/disabled in config
	if w.config.FaceSwap.Enabled != nil {
		w.supportsFaceSwap = *w.config.FaceSwap.Enabled
		if w.supportsFaceSwap {
			log.Printf("Face-swap support: ENABLED (explicit config)")
		} else {
			log.Printf("Face-swap support: DISABLED (explicit config)")
		}
		return
	}

	// Auto-detect based on:
	// 1. Face-swap model path exists
	// 2. inswapper_128.onnx model exists
	// 3. Sufficient VRAM (6GB+)

	// Check 1: Model path must be set
	modelPath := w.config.FaceSwap.ModelPath
	if modelPath == "" {
		w.supportsFaceSwap = false
		log.Printf("Face-swap support: DISABLED (FACESWAP_MODEL_PATH not set)")
		return
	}

	// Check 2: Verify the model directory exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		w.supportsFaceSwap = false
		log.Printf("Face-swap support: DISABLED (model directory not found at %s)", modelPath)
		return
	}

	// Check 3: Verify inswapper_128.onnx exists (required for face-swap)
	inswapperPath := filepath.Join(modelPath, "inswapper_128.onnx")
	if _, err := os.Stat(inswapperPath); os.IsNotExist(err) {
		w.supportsFaceSwap = false
		log.Printf("Face-swap support: DISABLED (inswapper_128.onnx not found at %s)", modelPath)
		log.Printf("  Tip: Download from https://huggingface.co/ezioruan/inswapper_128.onnx")
		return
	}

	// Check 4: Sufficient VRAM (6GB minimum for face-swap)
	if w.capabilities != nil && w.capabilities.VRAM < 6 {
		w.supportsFaceSwap = false
		log.Printf("Face-swap support: DISABLED (insufficient VRAM: %dGB < 6GB)", w.capabilities.VRAM)
		return
	}

	// All checks passed
	w.supportsFaceSwap = true
	log.Printf("Face-swap support: ENABLED (models found at %s)", modelPath)

	// Log enhancement capability
	if w.config.FaceSwap.Enhancement != nil && *w.config.FaceSwap.Enhancement {
		gfpganPath := filepath.Join(modelPath, "GFPGANv1.4.pth")
		if _, err := os.Stat(gfpganPath); os.IsNotExist(err) {
			log.Printf("  Note: GFPGAN not found - enhancement will be disabled")
		} else {
			log.Printf("  Enhancement: ENABLED (GFPGAN found)")
		}
	}
}

// canProcessFaceSwap checks if this worker can process face-swap jobs
func (w *Worker) canProcessFaceSwap() bool {
	return w.supportsFaceSwap
}
