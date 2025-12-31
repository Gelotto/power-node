package worker

import (
	"testing"
)

func TestDetermineServiceMode(t *testing.T) {
	tests := []struct {
		name       string
		computeCap string
		expected   string
	}{
		// GGUF mode cases (compute cap < 12.0)
		{"Ampere 8.6", "8.6", "gguf"},
		{"Ada Lovelace 8.9", "8.9", "gguf"},
		{"Hopper 9.0", "9.0", "gguf"},
		{"Pascal 6.1", "6.1", "gguf"},
		{"Turing 7.5", "7.5", "gguf"},

		// PyTorch mode cases (compute cap >= 12.0, Blackwell)
		{"Blackwell 12.0", "12.0", "pytorch"},
		{"Blackwell 12.5", "12.5", "pytorch"},
		{"Future 13.0", "13.0", "pytorch"},

		// Edge cases
		{"Empty string", "", "gguf"},
		{"Invalid format", "invalid", "gguf"},
		{"No decimal", "8", "gguf"},
		{"Major only 11", "11", "gguf"},
		{"Major only 12", "12", "pytorch"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := determineServiceMode(tt.computeCap)
			if got != tt.expected {
				t.Errorf("determineServiceMode(%q) = %q, want %q", tt.computeCap, got, tt.expected)
			}
		})
	}
}

func TestCalculateTierLimits_GGUF(t *testing.T) {
	tests := []struct {
		name          string
		vramGB        int
		wantTier      string
		wantMaxRes    int
		wantMaxSteps  int
	}{
		// Premium tier (16GB+)
		{"24GB GGUF", 24, "premium", 1536, 8},
		{"16GB GGUF", 16, "premium", 1536, 8},
		{"20GB GGUF", 20, "premium", 1536, 8},

		// Pro tier (12-15GB)
		{"15GB GGUF", 15, "pro", 1440, 8},
		{"12GB GGUF", 12, "pro", 1440, 8},
		{"14GB GGUF", 14, "pro", 1440, 8},

		// Basic tier (8-11GB)
		{"11GB GGUF", 11, "basic", 1024, 8},
		{"8GB GGUF", 8, "basic", 1024, 8},
		{"10GB GGUF", 10, "basic", 1024, 8},

		// Below minimum (<8GB)
		{"7GB GGUF", 7, "basic", 512, 4},
		{"6GB GGUF", 6, "basic", 512, 4},
		{"4GB GGUF", 4, "basic", 512, 4},
		{"0GB GGUF", 0, "basic", 512, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tier, maxRes, maxSteps := calculateTierLimits(tt.vramGB, "gguf")
			if tier != tt.wantTier {
				t.Errorf("calculateTierLimits(%d, gguf) tier = %q, want %q", tt.vramGB, tier, tt.wantTier)
			}
			if maxRes != tt.wantMaxRes {
				t.Errorf("calculateTierLimits(%d, gguf) maxRes = %d, want %d", tt.vramGB, maxRes, tt.wantMaxRes)
			}
			if maxSteps != tt.wantMaxSteps {
				t.Errorf("calculateTierLimits(%d, gguf) maxSteps = %d, want %d", tt.vramGB, maxSteps, tt.wantMaxSteps)
			}
		})
	}
}

func TestCalculateTierLimits_PyTorch(t *testing.T) {
	tests := []struct {
		name          string
		vramGB        int
		wantTier      string
		wantMaxRes    int
		wantMaxSteps  int
	}{
		// Premium tier (16GB+ for PyTorch)
		{"32GB PyTorch", 32, "premium", 1536, 8},
		{"24GB PyTorch", 24, "premium", 1536, 8},
		{"16GB PyTorch", 16, "premium", 1536, 8},

		// Pro tier (14-15GB for PyTorch)
		{"15GB PyTorch", 15, "pro", 1440, 8},
		{"14GB PyTorch", 14, "pro", 1440, 8},

		// Basic tier (<14GB for PyTorch)
		{"13GB PyTorch", 13, "basic", 1024, 8},
		{"12GB PyTorch", 12, "basic", 1024, 8},
		{"8GB PyTorch", 8, "basic", 1024, 8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tier, maxRes, maxSteps := calculateTierLimits(tt.vramGB, "pytorch")
			if tier != tt.wantTier {
				t.Errorf("calculateTierLimits(%d, pytorch) tier = %q, want %q", tt.vramGB, tier, tt.wantTier)
			}
			if maxRes != tt.wantMaxRes {
				t.Errorf("calculateTierLimits(%d, pytorch) maxRes = %d, want %d", tt.vramGB, maxRes, tt.wantMaxRes)
			}
			if maxSteps != tt.wantMaxSteps {
				t.Errorf("calculateTierLimits(%d, pytorch) maxSteps = %d, want %d", tt.vramGB, maxSteps, tt.wantMaxSteps)
			}
		})
	}
}

func TestCalculateTierLimits_VRAMBoundaries(t *testing.T) {
	// Test exact boundary values for GGUF mode
	tests := []struct {
		vram     int
		wantTier string
	}{
		// GGUF boundaries
		{7, "basic"},  // Below 8GB
		{8, "basic"},  // Exactly 8GB
		{11, "basic"}, // Below 12GB
		{12, "pro"},   // Exactly 12GB
		{15, "pro"},   // Below 16GB
		{16, "premium"}, // Exactly 16GB
	}

	for _, tt := range tests {
		tier, _, _ := calculateTierLimits(tt.vram, "gguf")
		if tier != tt.wantTier {
			t.Errorf("GGUF boundary: calculateTierLimits(%d, gguf) = %q, want %q", tt.vram, tier, tt.wantTier)
		}
	}

	// Test exact boundary values for PyTorch mode
	pytorchTests := []struct {
		vram     int
		wantTier string
	}{
		{13, "basic"},   // Below 14GB
		{14, "pro"},     // Exactly 14GB
		{15, "pro"},     // Below 16GB
		{16, "premium"}, // Exactly 16GB
	}

	for _, tt := range pytorchTests {
		tier, _, _ := calculateTierLimits(tt.vram, "pytorch")
		if tier != tt.wantTier {
			t.Errorf("PyTorch boundary: calculateTierLimits(%d, pytorch) = %q, want %q", tt.vram, tier, tt.wantTier)
		}
	}
}

func TestGPUCapabilities_String(t *testing.T) {
	caps := &GPUCapabilities{
		GPUModel:       "NVIDIA RTX 4090",
		VRAM:           24,
		ComputeCap:     "8.9",
		ServiceMode:    "gguf",
		CapabilityTier: "premium",
		MaxResolution:  1536,
		MaxSteps:       8,
	}

	str := caps.String()

	// Verify key components are in the string
	expectedParts := []string{
		"NVIDIA RTX 4090",
		"24GB VRAM",
		"CC 8.9",
		"gguf mode",
		"premium tier",
		"1536x1536",
		"8 steps",
	}

	for _, part := range expectedParts {
		if !containsSubstring(str, part) {
			t.Errorf("GPUCapabilities.String() = %q, missing %q", str, part)
		}
	}
}

func TestGPUCapabilities_String_PyTorch(t *testing.T) {
	caps := &GPUCapabilities{
		GPUModel:       "NVIDIA RTX 5090",
		VRAM:           32,
		ComputeCap:     "12.0",
		ServiceMode:    "pytorch",
		CapabilityTier: "premium",
		MaxResolution:  1536,
		MaxSteps:       8,
	}

	str := caps.String()

	if !containsSubstring(str, "pytorch mode") {
		t.Errorf("GPUCapabilities.String() = %q, should contain 'pytorch mode'", str)
	}
	if !containsSubstring(str, "CC 12.0") {
		t.Errorf("GPUCapabilities.String() = %q, should contain 'CC 12.0'", str)
	}
}

func TestGPUCapabilities_Struct(t *testing.T) {
	// Test struct field types and JSON tags
	caps := GPUCapabilities{
		GPUModel:       "Test GPU",
		VRAM:           16,
		ComputeCap:     "8.9",
		ServiceMode:    "gguf",
		MaxResolution:  1536,
		MaxSteps:       8,
		CapabilityTier: "premium",
	}

	// Verify fields are accessible and correctly typed
	if caps.GPUModel != "Test GPU" {
		t.Errorf("GPUModel = %q, want %q", caps.GPUModel, "Test GPU")
	}
	if caps.VRAM != 16 {
		t.Errorf("VRAM = %d, want %d", caps.VRAM, 16)
	}
	if caps.ComputeCap != "8.9" {
		t.Errorf("ComputeCap = %q, want %q", caps.ComputeCap, "8.9")
	}
	if caps.ServiceMode != "gguf" {
		t.Errorf("ServiceMode = %q, want %q", caps.ServiceMode, "gguf")
	}
	if caps.MaxResolution != 1536 {
		t.Errorf("MaxResolution = %d, want %d", caps.MaxResolution, 1536)
	}
	if caps.MaxSteps != 8 {
		t.Errorf("MaxSteps = %d, want %d", caps.MaxSteps, 8)
	}
	if caps.CapabilityTier != "premium" {
		t.Errorf("CapabilityTier = %q, want %q", caps.CapabilityTier, "premium")
	}
}

// Test realistic GPU configurations
func TestCalculateTierLimits_RealGPUs(t *testing.T) {
	tests := []struct {
		gpuName     string
		vramGB      int
		serviceMode string
		wantTier    string
	}{
		// Consumer GPUs - GGUF mode
		{"RTX 4090", 24, "gguf", "premium"},
		{"RTX 4080", 16, "gguf", "premium"},
		{"RTX 4070 Ti Super", 16, "gguf", "premium"},
		{"RTX 4070 Ti", 12, "gguf", "pro"},
		{"RTX 4070", 12, "gguf", "pro"},
		{"RTX 4060 Ti 16GB", 16, "gguf", "premium"},
		{"RTX 4060 Ti 8GB", 8, "gguf", "basic"},
		{"RTX 4060", 8, "gguf", "basic"},
		{"RTX 3090", 24, "gguf", "premium"},
		{"RTX 3080", 10, "gguf", "basic"},
		{"RTX 3070", 8, "gguf", "basic"},
		{"RTX 3060", 12, "gguf", "pro"},

		// Blackwell GPUs - PyTorch mode (compute cap >= 12.0)
		{"RTX 5090", 32, "pytorch", "premium"},
		{"RTX 5080", 16, "pytorch", "premium"},
		{"RTX 5070 Ti", 16, "pytorch", "premium"},
		{"RTX 5070", 12, "pytorch", "basic"}, // Only 12GB, so basic in PyTorch mode

		// Professional GPUs
		{"A100", 40, "gguf", "premium"},
		{"A100", 80, "gguf", "premium"},
		{"H100", 80, "pytorch", "premium"},
		{"L40S", 48, "gguf", "premium"},
	}

	for _, tt := range tests {
		t.Run(tt.gpuName, func(t *testing.T) {
			tier, _, _ := calculateTierLimits(tt.vramGB, tt.serviceMode)
			if tier != tt.wantTier {
				t.Errorf("%s (%dGB, %s): got tier %q, want %q",
					tt.gpuName, tt.vramGB, tt.serviceMode, tier, tt.wantTier)
			}
		})
	}
}

// Test service mode selection for real compute capabilities
func TestDetermineServiceMode_RealGPUs(t *testing.T) {
	tests := []struct {
		gpuName    string
		computeCap string
		wantMode   string
	}{
		// Ampere (sm_86)
		{"RTX 3090", "8.6", "gguf"},
		{"RTX 3080", "8.6", "gguf"},
		{"RTX 3070", "8.6", "gguf"},
		{"RTX 3060", "8.6", "gguf"},
		{"A100", "8.0", "gguf"},

		// Ada Lovelace (sm_89)
		{"RTX 4090", "8.9", "gguf"},
		{"RTX 4080", "8.9", "gguf"},
		{"RTX 4070", "8.9", "gguf"},
		{"RTX 4060", "8.9", "gguf"},
		{"L40S", "8.9", "gguf"},

		// Hopper (sm_90)
		{"H100", "9.0", "gguf"},

		// Blackwell (sm_120) - requires PyTorch
		{"RTX 5090", "12.0", "pytorch"},
		{"RTX 5080", "12.0", "pytorch"},
		{"RTX 5070 Ti", "12.0", "pytorch"},
		{"RTX 5070", "12.0", "pytorch"},
		{"B100", "12.0", "pytorch"},
	}

	for _, tt := range tests {
		t.Run(tt.gpuName, func(t *testing.T) {
			mode := determineServiceMode(tt.computeCap)
			if mode != tt.wantMode {
				t.Errorf("%s (CC %s): got mode %q, want %q",
					tt.gpuName, tt.computeCap, mode, tt.wantMode)
			}
		})
	}
}

// Helper function
func containsSubstring(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && (containsSubstringHelper(s, substr)))
}

func containsSubstringHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
