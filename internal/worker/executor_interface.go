package worker

import (
	"context"

	"github.com/Gelotto/power-node/internal/executor"
	"github.com/Gelotto/power-node/internal/models"
)

// Executor defines the interface for the Python executor
// This allows injection of mock executors for testing
type Executor interface {
	Start(pythonExec, scriptPath string, args []string, env map[string]string) error
	Stop() error
	Generate(ctx context.Context, req *models.GenerateRequest) (*models.GenerateResponse, error)
	GenerateVideo(ctx context.Context, req *models.GenerateVideoRequest) (*models.GenerateVideoResponse, error)
	GenerateFaceSwap(ctx context.Context, req *models.FaceSwapRequest) (*models.FaceSwapResponse, error)
	SetProgressCallback(cb executor.ProgressCallback)
}
