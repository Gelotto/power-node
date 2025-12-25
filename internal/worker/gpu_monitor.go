package worker

import (
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

// GPUMetrics contains current GPU state
type GPUMetrics struct {
	Utilization int // GPU utilization percentage (0-100)
	MemoryUsed  int // Memory used in MB
	MemoryTotal int // Total memory in MB
	Temperature int // Temperature in Celsius
}

// GPUMonitor monitors GPU usage and controls worker pausing
type GPUMonitor struct {
	config IdleConfig
	device nvml.Device

	mu          sync.RWMutex
	paused      bool
	lastMetrics GPUMetrics

	// Channels for signaling worker
	pauseChan  chan struct{}
	resumeChan chan struct{}
	stopChan   chan struct{}
}

// NewGPUMonitor creates a new GPU monitor
func NewGPUMonitor(config IdleConfig) (*GPUMonitor, error) {
	// Initialize NVML
	ret := nvml.Init()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("failed to initialize NVML: %v", nvml.ErrorString(ret))
	}

	// Get first GPU device
	device, ret := nvml.DeviceGetHandleByIndex(0)
	if ret != nvml.SUCCESS {
		nvml.Shutdown()
		return nil, fmt.Errorf("failed to get GPU device: %v", nvml.ErrorString(ret))
	}

	return &GPUMonitor{
		config:     config,
		device:     device,
		pauseChan:  make(chan struct{}, 1),
		resumeChan: make(chan struct{}, 1),
		stopChan:   make(chan struct{}),
	}, nil
}

// Start begins the monitoring loop
func (m *GPUMonitor) Start() {
	go m.monitorLoop()
}

// Stop stops the monitoring loop
func (m *GPUMonitor) Stop() {
	close(m.stopChan)
	nvml.Shutdown()
}

// IsPaused returns whether the worker should be paused
func (m *GPUMonitor) IsPaused() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.paused
}

// GetMetrics returns the latest GPU metrics
func (m *GPUMonitor) GetMetrics() GPUMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.lastMetrics
}

// PauseChan returns the channel that signals when to pause
func (m *GPUMonitor) PauseChan() <-chan struct{} {
	return m.pauseChan
}

// ResumeChan returns the channel that signals when to resume
func (m *GPUMonitor) ResumeChan() <-chan struct{} {
	return m.resumeChan
}

func (m *GPUMonitor) monitorLoop() {
	ticker := time.NewTicker(m.config.CheckInterval)
	defer ticker.Stop()

	var cooldownStart time.Time
	inCooldown := false

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			metrics := m.collectMetrics()
			m.mu.Lock()
			m.lastMetrics = metrics
			wasPaused := m.paused
			m.mu.Unlock()

			isBusy := m.isGPUBusy(metrics)

			if !wasPaused && isBusy {
				// GPU became busy - pause immediately
				m.mu.Lock()
				m.paused = true
				m.mu.Unlock()
				inCooldown = false

				log.Printf("GPU busy (util=%d%%, mem=%d/%dMB) - pausing job claiming",
					metrics.Utilization, metrics.MemoryUsed, metrics.MemoryTotal)

				// Non-blocking send
				select {
				case m.pauseChan <- struct{}{}:
				default:
				}

			} else if wasPaused && !isBusy {
				// GPU is idle - start or continue cooldown
				if !inCooldown {
					cooldownStart = time.Now()
					inCooldown = true
					log.Printf("GPU idle - starting cooldown (%v)", m.config.CooldownPeriod)
				} else if time.Since(cooldownStart) >= m.config.CooldownPeriod {
					// Cooldown complete - resume
					m.mu.Lock()
					m.paused = false
					m.mu.Unlock()
					inCooldown = false

					log.Printf("GPU idle for %v - resuming job claiming", m.config.CooldownPeriod)

					select {
					case m.resumeChan <- struct{}{}:
					default:
					}
				}
			} else if wasPaused && isBusy {
				// Still busy - reset cooldown
				inCooldown = false
			}
		}
	}
}

func (m *GPUMonitor) collectMetrics() GPUMetrics {
	var metrics GPUMetrics

	// Get utilization
	util, ret := m.device.GetUtilizationRates()
	if ret == nvml.SUCCESS {
		metrics.Utilization = int(util.Gpu)
	}

	// Get memory info
	mem, ret := m.device.GetMemoryInfo()
	if ret == nvml.SUCCESS {
		metrics.MemoryUsed = int(mem.Used / (1024 * 1024))  // Convert to MB
		metrics.MemoryTotal = int(mem.Total / (1024 * 1024)) // Convert to MB
	}

	// Get temperature
	temp, ret := m.device.GetTemperature(nvml.TEMPERATURE_GPU)
	if ret == nvml.SUCCESS {
		metrics.Temperature = int(temp)
	}

	return metrics
}

func (m *GPUMonitor) isGPUBusy(metrics GPUMetrics) bool {
	// Check utilization threshold
	if metrics.Utilization > m.config.UtilizationThreshold {
		return true
	}

	// Check memory threshold
	if metrics.MemoryTotal > 0 {
		memPercent := (metrics.MemoryUsed * 100) / metrics.MemoryTotal
		if memPercent > m.config.MemoryThreshold {
			return true
		}
	}

	return false
}
