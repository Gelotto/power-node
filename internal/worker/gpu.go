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
// MaxResolution values are capped to Z-Image-Turbo's actual limits (~2MP max)
func calculateTierLimits(vramGB int, serviceMode string) (tier string, maxRes, maxSteps int) {
	// PyTorch mode requires more VRAM, so adjust thresholds
	if serviceMode == "pytorch" {
		// PyTorch mode - higher VRAM requirements
		switch {
		case vramGB >= 16:
			return "premium", 1536, 8 // Z-Image-Turbo: ~2MP max, 8 NFE optimal
		case vramGB >= 14:
			return "pro", 1440, 8
		default:
			return "basic", 1024, 8
		}
	}

	// GGUF mode - more efficient memory usage
	switch {
	case vramGB >= 16:
		return "premium", 1536, 8 // Z-Image-Turbo: ~2MP max, 8 NFE optimal
	case vramGB >= 12:
		return "pro", 1440, 8
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

// SupportableModel describes a model that can run on this GPU
type SupportableModel struct {
	Name         string `json:"name"`
	CanRunGGUF   bool   `json:"can_run_gguf"`
	CanRunPyTorch bool  `json:"can_run_pytorch"`
	MinVRAM      int    `json:"min_vram"` // Minimum VRAM required
}

// DetermineSupportableModels returns which image models this GPU can run
// based on VRAM and compute capability
func DetermineSupportableModels(vramGB int, computeCap string) []SupportableModel {
	models := []SupportableModel{}
	isBlackwell := isBlackwellGPU(computeCap)

	// Z-Image-Turbo requirements:
	// - GGUF mode: 8GB+ (for non-Blackwell GPUs)
	// - PyTorch mode: 14GB+ (for any GPU)
	zimage := SupportableModel{
		Name:    "z-image-turbo",
		MinVRAM: 8,
	}
	if !isBlackwell && vramGB >= 8 {
		zimage.CanRunGGUF = true
	}
	if vramGB >= 14 {
		zimage.CanRunPyTorch = true
	}
	if zimage.CanRunGGUF || zimage.CanRunPyTorch {
		models = append(models, zimage)
	}

	// FLUX.1-schnell requirements:
	// - GGUF mode: 12GB+ (for non-Blackwell GPUs, Q8_0 quantization)
	// - PyTorch mode: 16GB+ (for any GPU)
	flux := SupportableModel{
		Name:    "flux-schnell",
		MinVRAM: 12,
	}
	if !isBlackwell && vramGB >= 12 {
		flux.CanRunGGUF = true
	}
	if vramGB >= 16 {
		flux.CanRunPyTorch = true
	}
	if flux.CanRunGGUF || flux.CanRunPyTorch {
		models = append(models, flux)
	}

	return models
}

// isBlackwellGPU checks if the GPU is a Blackwell architecture (compute cap 12.0+)
func isBlackwellGPU(computeCap string) bool {
	parts := strings.Split(computeCap, ".")
	if len(parts) < 1 {
		return false
	}
	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return false
	}
	return major >= 12
}

// GetSupportableModelNames returns just the names of supportable models
func GetSupportableModelNames(vramGB int, computeCap string) []string {
	models := DetermineSupportableModels(vramGB, computeCap)
	names := make([]string, 0, len(models))
	for _, m := range models {
		names = append(names, m.Name)
	}
	return names
}

// CanRunModel checks if a specific model can run on this GPU
func CanRunModel(vramGB int, computeCap, modelName string) bool {
	models := DetermineSupportableModels(vramGB, computeCap)
	for _, m := range models {
		if m.Name == modelName {
			return true
		}
	}
	return false
}
