# High-Performance GEMM Makefile

NVCC = nvcc
CUDA_ARCH ?= 89
NVCC_ARCH_FLAGS = -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
                  -gencode arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH)
NVCC_FLAGS = -O3 -std=c++17 --use_fast_math $(NVCC_ARCH_FLAGS)
DEBUG_FLAGS = -g -G -lineinfo

# 目录
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

# 源文件
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# 目标
TARGET = benchmark

.PHONY: all clean debug profile

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SRC_DIR)/benchmark.cu $(SRC_DIR)/v1_naive.cu $(SRC_DIR)/v2_tiled.cu
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) $(SRC_DIR)/benchmark.cu -o $@ -lcublas

# 调试版本
debug: NVCC_FLAGS = $(DEBUG_FLAGS)
debug: clean all

# Profile 版本
profile: NVCC_FLAGS += -lineinfo
profile: clean all
	ncu --set full ./$(TARGET)

# 单独编译测试
test_v1: $(SRC_DIR)/v1_naive.cu
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) $< -o test_v1 -DTEST_MAIN

test_v2: $(SRC_DIR)/v2_tiled.cu
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) $< -o test_v2 -DTEST_MAIN

clean:
	rm -rf $(BUILD_DIR) $(TARGET) test_v*

# 帮助
help:
	@echo "使用方法:"
	@echo "  make        - 编译所有版本"
	@echo "  make debug  - 编译调试版本"
	@echo "  make profile- 编译并运行 Nsight Compute"
	@echo "  make test_v1- 单独测试 V1"
	@echo "  make test_v2- 单独测试 V2"
	@echo "  make clean  - 清理编译文件"
	@echo "  例: make CUDA_ARCH=89"
	@echo "  例: make CUDA_ARCH=120 (需 nvcc 支持)"
