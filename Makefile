# Bio-Faithful CUDA Neural Network - Simple Makefile
# =================================================

# CUDA Configuration (Fixed paths for your system)
NVCC := /usr/bin/nvcc
CXX := g++

# Compiler flags (Simplified for compatibility)
NVCC_FLAGS := -O3 -arch=sm_75
CXX_FLAGS := -O3 -Wall
LIBS := -lcudart -lcurand

# Targets
TARGET := model
SOURCE := model.cu

# Build rules
all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "🚀 Building Bio-Faithful CUDA Neural Network (Simple)..."
	@echo "⚡ Target GPU: GTX 1650 (sm_75)"
	$(NVCC) $(NVCC_FLAGS) $(SOURCE) -o $(TARGET) $(LIBS)
	@echo "✅ Build complete: $(TARGET)"
	@echo "🧠 Ready for ultra-high performance brain simulation!"

# Run the simulation
run: $(TARGET)
	@echo "🚀 Launching Bio-Faithful CUDA Neural Network..."
	@echo "⚡ Preparing for brain simulation at light speed..."
	./$(TARGET)

# Performance test
perf: $(TARGET)
	@echo "📊 Performance test..."
	time ./$(TARGET)

# Check GPU
check-gpu:
	@echo "🔍 GPU Information:"
	nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits

# Debug build
debug: NVCC_FLAGS := -g -G -arch=sm_75
debug: $(TARGET)
	@echo "🐛 Debug build complete"

# Clean
clean:
	@echo "🧹 Cleaning..."
	rm -f $(TARGET) *.o
	@echo "✅ Clean complete"

# Help
help:
	@echo "🧠 Bio-Faithful CUDA Neural Network - Simple Build"
	@echo "=================================================="
	@echo ""
	@echo "📋 Available targets:"
	@echo "  make          - Build the CUDA neural network"
	@echo "  make run      - Build and run simulation"
	@echo "  make perf     - Performance benchmark"
	@echo "  make debug    - Debug build"
	@echo "  make check-gpu- Check GPU info"
	@echo "  make clean    - Clean build files"
	@echo "  make help     - Show this help"
	@echo ""
	@echo "🚀 Quick start: make run"

.PHONY: all run perf debug check-gpu clean help
