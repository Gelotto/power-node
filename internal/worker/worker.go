package worker

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
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

	if w.config.NeedsRegistration() {
		return fmt.Errorf("worker not registered. Please register at https://gen.gelotto.io/workers/register and add your credentials to config.yaml")
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

	log.Println("Worker fully initialized!")

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

		case <-ticker.C:
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

// processJob processes a single job
func (w *Worker) processJob(ctx context.Context, job *models.Job) {
	log.Printf("Processing job %s...", job.ID)
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
		log.Printf("Generation failed for job %s: %v", job.ID, err)
		if err := w.apiClient.FailJob(ctx, w.id, job.ID, err.Error()); err != nil {
			log.Printf("Failed to report job failure: %v", err)
		}
		return
	}

	elapsed := time.Since(startTime)
	generationMs := int(elapsed.Milliseconds())
	log.Printf("Generation completed in %.2f seconds", elapsed.Seconds())

	log.Printf("Uploading result for job %s...", job.ID)
	if err := w.apiClient.CompleteJob(ctx, w.id, job.ID, result.ImageData, generationMs); err != nil {
		log.Printf("Failed to complete job %s: %v", job.ID, err)
		return
	}

	log.Printf("Job %s completed successfully!", job.ID)
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
			if err := w.apiClient.Heartbeat(ctx, w.id, "online", data); err != nil {
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

	return data
}

// Stop gracefully stops the worker
func (w *Worker) Stop() error {
	log.Println("Stopping worker...")

	close(w.stopChan)

	if w.pythonExec != nil {
		if err := w.pythonExec.Stop(); err != nil {
			log.Printf("Error stopping Python executor: %v", err)
		}
	}

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
