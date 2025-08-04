# DUDUX AI - Makefile
# Bio-Faithful Neural Network CUDA Compilation
# ===========================================

# CUDA compiler
NVCC := /usr/bin/nvcc
CXX := g++

# CUDA flags for GTX 1650 (Compute Capability 7.5)
NVCC_FLAGS := -O3 -arch=sm_75 -use_fast_math
CXX_FLAGS := -O3 -Wall
LIBS := -lcudart -lcurand

# Targets
TRAINING_TARGET := model
INFERENCE_TARGET := inference
TRAINING_SRC := model.cu
INFERENCE_SRC := inference.cu

# Default target
all: $(TRAINING_TARGET) $(INFERENCE_TARGET)

# Training program
$(TRAINING_TARGET): $(TRAINING_SRC)
	@echo "� Compiling DUDUX AI Training System..."
	$(NVCC) $(NVCC_FLAGS) $(TRAINING_SRC) -o $(TRAINING_TARGET) $(LIBS)
	@echo "✅ Training program compiled: ./$(TRAINING_TARGET)"

# Inference program  
$(INFERENCE_TARGET): $(INFERENCE_SRC)
	@echo "🔨 Compiling DUDUX AI Inference System..."
	$(NVCC) $(NVCC_FLAGS) $(INFERENCE_SRC) -o $(INFERENCE_TARGET) $(LIBS)
	@echo "✅ Inference program compiled: ./$(INFERENCE_TARGET)"

# Train the model
train: $(TRAINING_TARGET)
	@echo "🚀 Starting training session..."
	./$(TRAINING_TARGET)

# Run inference
infer: $(INFERENCE_TARGET)
	@echo "🧠 Starting inference session..."
	./$(INFERENCE_TARGET)

# Show help
help:
	@echo "DUDUX AI - Makefile Commands"
	@echo "============================"
	@echo "make all       - Compile both training and inference"
	@echo "make model     - Compile training program only"
	@echo "make inference - Compile inference program only"
	@echo "make train     - Compile and run training"
	@echo "make infer     - Compile and run inference"
	@echo "make clean     - Remove compiled files"
	@echo "make help      - Show this help"
	@echo ""
	@echo "Usage:"
	@echo "  1. make train    # Train the model"
	@echo "  2. make infer    # Run interactive inference"

# Clean
clean:
	@echo "🧹 Cleaning..."
	rm -f $(TARGET) *.o
	@echo "✅ Clean complete"

.PHONY: all run perf debug check-gpu clean help
