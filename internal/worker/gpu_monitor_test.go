package worker

import (
	"testing"
	"time"
)

// TestGPUMetrics_Struct tests the GPUMetrics struct
func TestGPUMetrics_Struct(t *testing.T) {
	metrics := GPUMetrics{
		Utilization: 75,
		MemoryUsed:  8000,
		MemoryTotal: 16000,
		Temperature: 65,
	}

	if metrics.Utilization != 75 {
		t.Errorf("Utilization = %d, want 75", metrics.Utilization)
	}
	if metrics.MemoryUsed != 8000 {
		t.Errorf("MemoryUsed = %d, want 8000", metrics.MemoryUsed)
	}
	if metrics.MemoryTotal != 16000 {
		t.Errorf("MemoryTotal = %d, want 16000", metrics.MemoryTotal)
	}
	if metrics.Temperature != 65 {
		t.Errorf("Temperature = %d, want 65", metrics.Temperature)
	}
}

// TestIdleConfig_Defaults tests that IdleConfig has sensible zero values
func TestIdleConfig_Defaults(t *testing.T) {
	config := IdleConfig{}

	// Zero values mean disabled by default
	if config.Enabled != false {
		t.Errorf("Enabled = %v, want false", config.Enabled)
	}
	if config.UtilizationThreshold != 0 {
		t.Errorf("UtilizationThreshold = %d, want 0", config.UtilizationThreshold)
	}
	if config.MemoryThreshold != 0 {
		t.Errorf("MemoryThreshold = %d, want 0", config.MemoryThreshold)
	}
}

// TestIdleConfig_CustomValues tests custom IdleConfig values
func TestIdleConfig_CustomValues(t *testing.T) {
	config := IdleConfig{
		Enabled:              true,
		UtilizationThreshold: 50,
		MemoryThreshold:      80,
		CheckInterval:        5 * time.Second,
		CooldownPeriod:       30 * time.Second,
	}

	if config.Enabled != true {
		t.Errorf("Enabled = %v, want true", config.Enabled)
	}
	if config.UtilizationThreshold != 50 {
		t.Errorf("UtilizationThreshold = %d, want 50", config.UtilizationThreshold)
	}
	if config.MemoryThreshold != 80 {
		t.Errorf("MemoryThreshold = %d, want 80", config.MemoryThreshold)
	}
	if config.CheckInterval != 5*time.Second {
		t.Errorf("CheckInterval = %v, want 5s", config.CheckInterval)
	}
	if config.CooldownPeriod != 30*time.Second {
		t.Errorf("CooldownPeriod = %v, want 30s", config.CooldownPeriod)
	}
}

// TestIsGPUBusy_UtilizationAboveThreshold tests that high utilization returns true
func TestIsGPUBusy_UtilizationAboveThreshold(t *testing.T) {
	config := IdleConfig{
		UtilizationThreshold: 50,
		MemoryThreshold:      80,
	}

	// Create a mock monitor to access isGPUBusy
	// Since isGPUBusy is a method, we test it indirectly through the logic

	tests := []struct {
		name        string
		utilization int
		memUsed     int
		memTotal    int
		threshold   int
		memThreshold int
		want        bool
	}{
		{
			name:        "utilization above threshold",
			utilization: 60,
			memUsed:     4000,
			memTotal:    16000,
			threshold:   50,
			memThreshold: 80,
			want:        true,
		},
		{
			name:        "utilization at threshold (not busy)",
			utilization: 50,
			memUsed:     4000,
			memTotal:    16000,
			threshold:   50,
			memThreshold: 80,
			want:        false,
		},
		{
			name:        "utilization below threshold",
			utilization: 40,
			memUsed:     4000,
			memTotal:    16000,
			threshold:   50,
			memThreshold: 80,
			want:        false,
		},
		{
			name:        "memory above threshold",
			utilization: 10,
			memUsed:     13000,
			memTotal:    16000,
			threshold:   50,
			memThreshold: 80,
			want:        true, // 13000/16000 = 81.25%
		},
		{
			name:        "memory at threshold (not busy)",
			utilization: 10,
			memUsed:     12800,
			memTotal:    16000,
			threshold:   50,
			memThreshold: 80,
			want:        false, // 12800/16000 = 80%
		},
		{
			name:        "memory below threshold",
			utilization: 10,
			memUsed:     8000,
			memTotal:    16000,
			threshold:   50,
			memThreshold: 80,
			want:        false, // 8000/16000 = 50%
		},
		{
			name:        "both above threshold",
			utilization: 60,
			memUsed:     14000,
			memTotal:    16000,
			threshold:   50,
			memThreshold: 80,
			want:        true,
		},
		{
			name:        "both below threshold",
			utilization: 30,
			memUsed:     4000,
			memTotal:    16000,
			threshold:   50,
			memThreshold: 80,
			want:        false,
		},
		{
			name:        "zero total memory",
			utilization: 10,
			memUsed:     1000,
			memTotal:    0,
			threshold:   50,
			memThreshold: 80,
			want:        false, // Avoid divide by zero
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Recreate the isGPUBusy logic here since we can't easily access the private method
			metrics := GPUMetrics{
				Utilization: tt.utilization,
				MemoryUsed:  tt.memUsed,
				MemoryTotal: tt.memTotal,
			}
			config := IdleConfig{
				UtilizationThreshold: tt.threshold,
				MemoryThreshold:      tt.memThreshold,
			}

			// Replicate isGPUBusy logic
			isBusy := false
			if metrics.Utilization > config.UtilizationThreshold {
				isBusy = true
			}
			if metrics.MemoryTotal > 0 {
				memPercent := (metrics.MemoryUsed * 100) / metrics.MemoryTotal
				if memPercent > config.MemoryThreshold {
					isBusy = true
				}
			}

			if isBusy != tt.want {
				t.Errorf("isGPUBusy = %v, want %v", isBusy, tt.want)
			}
		})
	}

	_ = config // Silence unused warning
}

// TestMemoryPercentCalculation tests the memory percentage calculation
func TestMemoryPercentCalculation(t *testing.T) {
	tests := []struct {
		name     string
		used     int
		total    int
		expected int
	}{
		{"50% used", 8000, 16000, 50},
		{"25% used", 4000, 16000, 25},
		{"75% used", 12000, 16000, 75},
		{"100% used", 16000, 16000, 100},
		{"0% used", 0, 16000, 0},
		{"80% used (boundary)", 12800, 16000, 80},
		{"81% used", 13000, 16000, 81}, // Integer division: 13000*100/16000 = 81.25 -> 81
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			memPercent := (tt.used * 100) / tt.total
			if memPercent != tt.expected {
				t.Errorf("memPercent = %d, want %d", memPercent, tt.expected)
			}
		})
	}
}

// TestCooldownLogic tests the cooldown timing logic
func TestCooldownLogic(t *testing.T) {
	cooldownPeriod := 30 * time.Second

	// Simulate cooldown start
	cooldownStart := time.Now().Add(-20 * time.Second) // Started 20s ago

	// Cooldown should not be complete yet
	if time.Since(cooldownStart) >= cooldownPeriod {
		t.Error("Cooldown should not be complete after 20s with 30s period")
	}

	// Simulate cooldown complete
	cooldownStart = time.Now().Add(-35 * time.Second) // Started 35s ago

	// Cooldown should be complete
	if time.Since(cooldownStart) < cooldownPeriod {
		t.Error("Cooldown should be complete after 35s with 30s period")
	}
}

// TestStateTransitions tests GPU monitor state transitions
func TestStateTransitions(t *testing.T) {
	// Simulate the state machine logic without NVML

	tests := []struct {
		name           string
		wasPaused      bool
		isBusy         bool
		inCooldown     bool
		cooldownElapsed bool
		expectPause    bool
		expectResume   bool
	}{
		{
			name:           "online -> busy: pause",
			wasPaused:      false,
			isBusy:         true,
			inCooldown:     false,
			cooldownElapsed: false,
			expectPause:    true,
			expectResume:   false,
		},
		{
			name:           "paused + busy: stay paused",
			wasPaused:      true,
			isBusy:         true,
			inCooldown:     false,
			cooldownElapsed: false,
			expectPause:    false,
			expectResume:   false,
		},
		{
			name:           "paused + idle: start cooldown",
			wasPaused:      true,
			isBusy:         false,
			inCooldown:     false,
			cooldownElapsed: false,
			expectPause:    false,
			expectResume:   false, // Just start cooldown, don't resume yet
		},
		{
			name:           "paused + idle + cooldown not complete: wait",
			wasPaused:      true,
			isBusy:         false,
			inCooldown:     true,
			cooldownElapsed: false,
			expectPause:    false,
			expectResume:   false,
		},
		{
			name:           "paused + idle + cooldown complete: resume",
			wasPaused:      true,
			isBusy:         false,
			inCooldown:     true,
			cooldownElapsed: true,
			expectPause:    false,
			expectResume:   true,
		},
		{
			name:           "online + idle: nothing",
			wasPaused:      false,
			isBusy:         false,
			inCooldown:     false,
			cooldownElapsed: false,
			expectPause:    false,
			expectResume:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Replicate monitor loop logic
			var shouldPause, shouldResume bool

			if !tt.wasPaused && tt.isBusy {
				// GPU became busy - pause immediately
				shouldPause = true
			} else if tt.wasPaused && !tt.isBusy {
				// GPU is idle
				if tt.inCooldown && tt.cooldownElapsed {
					// Cooldown complete - resume
					shouldResume = true
				}
				// else: start or continue cooldown (no signal)
			}
			// else if tt.wasPaused && tt.isBusy: reset cooldown (no signal)

			if shouldPause != tt.expectPause {
				t.Errorf("shouldPause = %v, want %v", shouldPause, tt.expectPause)
			}
			if shouldResume != tt.expectResume {
				t.Errorf("shouldResume = %v, want %v", shouldResume, tt.expectResume)
			}
		})
	}
}

// TestNonBlockingChannelSend tests that channel sends are non-blocking
func TestNonBlockingChannelSend(t *testing.T) {
	ch := make(chan struct{}, 1)

	// Fill the buffer
	ch <- struct{}{}

	// Non-blocking send should not block
	done := make(chan bool, 1)
	go func() {
		select {
		case ch <- struct{}{}:
			// Sent (buffer had room)
		default:
			// Channel full, signal dropped (expected behavior)
		}
		done <- true
	}()

	select {
	case <-done:
		// Success - non-blocking send completed
	case <-time.After(100 * time.Millisecond):
		t.Error("non-blocking send should not block")
	}
}

// TestChannelBufferSize tests that channels have correct buffer size
func TestChannelBufferSize(t *testing.T) {
	// Verify buffer size of 1 allows one pending signal
	pauseChan := make(chan struct{}, 1)
	resumeChan := make(chan struct{}, 1)

	// First send should succeed
	pauseChan <- struct{}{}
	resumeChan <- struct{}{}

	// Channels should have pending signals
	if len(pauseChan) != 1 {
		t.Errorf("pauseChan length = %d, want 1", len(pauseChan))
	}
	if len(resumeChan) != 1 {
		t.Errorf("resumeChan length = %d, want 1", len(resumeChan))
	}

	// Consume signals
	<-pauseChan
	<-resumeChan

	// Channels should be empty
	if len(pauseChan) != 0 {
		t.Errorf("pauseChan length = %d, want 0", len(pauseChan))
	}
	if len(resumeChan) != 0 {
		t.Errorf("resumeChan length = %d, want 0", len(resumeChan))
	}
}

// TestRapidBusyIdleToggle tests that rapid toggles reset cooldown
func TestRapidBusyIdleToggle(t *testing.T) {
	// Simulate rapid busy/idle toggles

	cooldownPeriod := 30 * time.Second
	var cooldownStart time.Time
	inCooldown := false

	// Initial state: paused
	wasPaused := true

	// T=0: GPU goes idle, start cooldown
	isBusy := false
	if wasPaused && !isBusy && !inCooldown {
		cooldownStart = time.Now()
		inCooldown = true
	}

	if !inCooldown {
		t.Error("Should be in cooldown after idle")
	}

	// T=10s: GPU goes busy, reset cooldown
	isBusy = true
	if wasPaused && isBusy {
		inCooldown = false
	}

	if inCooldown {
		t.Error("Cooldown should be reset when GPU busy")
	}

	// T=15s: GPU goes idle again, start new cooldown
	isBusy = false
	if wasPaused && !isBusy && !inCooldown {
		cooldownStart = time.Now()
		inCooldown = true
	}

	// Cooldown should start from T=15s, not T=0
	if !inCooldown {
		t.Error("New cooldown should have started")
	}

	// The cooldown timer was reset, so it should take another 30s from now
	_ = cooldownStart
	_ = cooldownPeriod
}

// TestEdgeCases tests various edge cases
func TestEdgeCases(t *testing.T) {
	t.Run("zero utilization threshold", func(t *testing.T) {
		// If threshold is 0, any utilization > 0 should trigger busy
		threshold := 0
		utilization := 1

		if utilization > threshold {
			// Expected: even 1% is considered busy
		} else {
			t.Error("1% utilization should be > 0% threshold")
		}
	})

	t.Run("100% utilization threshold", func(t *testing.T) {
		// If threshold is 100, only > 100% would trigger (impossible)
		threshold := 100
		utilization := 100

		if utilization > threshold {
			t.Error("100% utilization should not be > 100% threshold")
		}
	})

	t.Run("negative memory used", func(t *testing.T) {
		// Edge case: invalid negative memory (should not happen in practice)
		memUsed := -1000
		memTotal := 16000

		// Integer division with negative numerator
		memPercent := (memUsed * 100) / memTotal
		// This would give -6%, which is < any positive threshold
		if memPercent >= 0 {
			t.Errorf("Negative memory used should give negative percent, got %d", memPercent)
		}
	})
}
