# Adapted from https://github.com/TravisWThompson1/Makefile_Example_CUDA_CPP_To_Executable
# Windows-only, using Visual Studio toolchain

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=D:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=cl.exe
CC_FLAGS=-g -O0
CC_LIBS=

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-dc -G -g -O0
NVCC_LIBS=

# CUDA include directory:
CUDA_INC_DIR=/I"$(CUDA_ROOT_DIR)/include"
# CUDA linking libraries:
CUDA_LINK_LIBS="$(CUDA_ROOT_DIR)/lib/x64/cudart.lib"

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

## Make variables ##

# Target executable name:
EXE = parallellines

# Object files:
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.obj,$(wildcard $(SRC_DIR)/*.cpp)) \
       $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.obj,$(wildcard $(SRC_DIR)/*.cu))

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(NVCC) $(OBJS) $(CUDA_LINK_LIBS) -o $@

# Force recompilation of header files by deleting main's object before make
remake : clean-main $(EXE)

# Compile main.cu file to object files:
$(OBJ_DIR)/%.obj : $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.obj : $(SRC_DIR)/%.cpp $(INC_DIR)/%.hpp
	$(CC) $(CC_FLAGS) -c $< /Fo$@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.obj : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean object from main only
clean-main:
	del /q $(OBJ_DIR)\\main.obj

# Clean objects in object directory.
clean:
	del /s /q $(OBJ_DIR)

run : $(EXE)
	$(EXE)

remake-run : remake
	$(EXE)