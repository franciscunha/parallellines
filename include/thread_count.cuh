#pragma once

#include <cmath>

#define TARGET_GRID_SIZE 16 // value depends on GPU, check NVIDIA Nsight compute

inline void calculate_kernel_size(size_t target_num_threads, size_t *out_num_blocks, size_t *out_num_threads_per_block)
{
    *out_num_threads_per_block = std::ceil(target_num_threads / (float)TARGET_GRID_SIZE);
    size_t remainder = (*out_num_threads_per_block) % 32;
    *out_num_threads_per_block = (*out_num_threads_per_block) + (remainder == 0 ? 0 : (32 - remainder));
    *out_num_blocks = std::ceil(target_num_threads / (float)(*out_num_threads_per_block));
}

