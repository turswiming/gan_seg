#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <cmath>

// Reduced from 8192 to 4096 to avoid shared memory limit on RTX 5090 (compute capability 12.0)
// RTX 5090 has 48KB shared memory limit, and 8192 threads would require 64KB
#define TOTAL_THREADS 4096
#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, TOTAL_THREADS), 1);
}
#endif
