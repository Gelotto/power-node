package worker

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// GPUCapabilities holds detected GPU information
type GPUCapabilities struct {
	GPUModel      string `json:"gpu_model"`
	VRAM          int    `json:"vram"`           // GB
	ComputeCap    string `json:"compute_cap"`    // e.g., "8.9"
	ServiceMode   string `json:"service_mode"`   // "gguf" or "pytorch"
	MaxResolution int    `json:"max_resolution"` // Maximum resolution dimension
	MaxSteps      int    `json:"max_steps"`      // Maximum inference steps
	CapabilityTier string `json:"capability_tier"` // "basic", "pro", "premium"
}

// DetectGPUCapabilities detects GPU capabilities using nvidia-smi
func DetectGPUCapabilities() (*GPUCapabilities, error) {
	// Query nvidia-smi for GPU name, memory, and compute capability
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=name,memory.total,compute_cap",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi not available or failed: %w", err)
	}

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 || lines[0] == "" {
		return nil, fmt.Errorf("no GPU detected")
	}

	// Parse first GPU (use primary GPU if multiple)
	parts := strings.Split(lines[0], ", ")
	if len(parts) < 3 {
		return nil, fmt.Errorf("unexpected nvidia-smi output format: %s", lines[0])
	}

	gpuName := strings.TrimSpace(parts[0])
	memoryMB, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if err != nil {
		return nil, fmt.Errorf("failed to parse memory: %w", err)
	}
	computeCap := strings.TrimSpace(parts[2])

	// Convert MB to GB (nvidia-smi returns memory.total in MiB)
	vramGB := int(memoryMB / 1024)

	caps := &GPUCapabilities{
		GPUModel:   gpuName,
		VRAM:       vramGB,
		ComputeCap: computeCap,
	}

	// Determine service mode based on compute capability
	// Blackwell (sm_120+, compute cap 12.0+) requires PyTorch due to GGUF/ggml kernel issues
	caps.ServiceMode = determineServiceMode(computeCap)

	// Calculate capability tier and limits based on VRAM
	caps.CapabilityTier, caps.MaxResolution, caps.MaxSteps = calculateTierLimits(vramGB, caps.ServiceMode)

	return caps, nil
}

// determineServiceMode determines if GGUF or PyTorch should be used
func determineServiceMode(computeCap string) string {
	// Parse compute capability (e.g., "8.9", "12.0")
	parts := strings.Split(computeCap, ".")
	if len(parts) < 1 {
		return "gguf" // Default to GGUF
	}

	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return "gguf"
	}

	// Blackwell GPUs (RTX 50-series) have compute capability 12.0+
	// GGUF/ggml kernels don't support sm_120, so we use PyTorch
	if major >= 12 {
		return "pytorch"
	}

	return "gguf"
}

// calculateTierLimits determines capability tier and limits based on VRAM
func calculateTierLimits(vramGB int, serviceMode string) (tier string, maxRes, maxSteps int) {
	// PyTorch mode requires more VRAM, so adjust thresholds
	if serviceMode == "pytorch" {
		// PyTorch mode - higher VRAM requirements
		switch {
		case vramGB >= 16:
			return "premium", 4096, 32
		case vramGB >= 14:
			return "pro", 2048, 16
		default:
			return "basic", 1024, 8
		}
	}

	// GGUF mode - more efficient memory usage
	switch {
	case vramGB >= 16:
		return "premium", 4096, 32
	case vramGB >= 12:
		return "pro", 2048, 16
	case vramGB >= 8:
		return "basic", 1024, 8
	default:
		// Below minimum, but still allow basic
		return "basic", 512, 4
	}
}

// String returns a human-readable summary of GPU capabilities
func (c *GPUCapabilities) String() string {
	return fmt.Sprintf("%s (%dGB VRAM, CC %s, %s mode, %s tier, max %dx%d @ %d steps)",
		c.GPUModel, c.VRAM, c.ComputeCap, c.ServiceMode,
		c.CapabilityTier, c.MaxResolution, c.MaxResolution, c.MaxSteps)
}
