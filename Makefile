# Adapted from https://github.com/TravisWThompson1/Makefile_Example_CUDA_CPP_To_Executable
# Windows-only, using Visual Studio toolchain

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=D:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=cl.exe
CC_FLAGS=
CC_LIBS=

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=
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
	$(CC) $(OBJS) /Fe$@ $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.obj : $(SRC_DIR)/%.cpp
	$(CC) $(CC_FLAGS) -c $< /Fo$@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.obj : $(SRC_DIR)/%.cpp $(INC_DIR)/%.hpp
	$(CC) $(CC_FLAGS) -c $< /Fo$@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.obj : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean objects in object directory.
clean:
	del /s /q $(OBJ_DIR)
