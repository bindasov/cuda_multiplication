#include <cstdlib>
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// size_type must be unsigned
typedef unsigned long long size_type;

const size_type MOD = 5728409;
const size_type ROOT = 2983;
const size_type ROOT_ORDER = 1 << 3;

// бинарное возведение в степень по модулю (a^n(mod))
size_type binary_pow(size_type a, size_type n, size_type mod) {
    size_type res = 1;
    while (n) {
        if (n & size_type(1))
            res = res * a % mod;
        a = a * a % mod;
        n >>= 1;
    }
    return res;
}

// находит обратный элемент как n^(mod-2)
size_type reverse(size_type n, size_type mod) {
    return binary_pow(n, mod - 2, mod);
}

void print_array(const std::string &name, const size_type *vec, size_type size, bool polynomial) {
    std::cout << name << ": ";
    for (auto i = 0; i < size; i++) {
        if (polynomial) {
            if (vec[i] != 0)
                std::cout << vec[i] << "x^" << i << " ";
        } else {
            if (i != size - 1)
                std::cout << vec[i] << ", ";
            else
                std::cout << vec[i];
        }
    }
    std::cout << std::endl;
}

void check_error(cudaError_t err) {
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "error code: " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void bit_reversal(size_type *vec, size_type size, size_type sizeLog2) {
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        auto bitsNumber = sizeof(i) * 8;
        size_type reverse_i = 0;
        for (auto j = 0; j < bitsNumber; j++)
            if ((i & (size_type(1) << j)))
                reverse_i |= size_type(1) << (bitsNumber - 1 - j);
        reverse_i >>= (bitsNumber - sizeLog2);
        if (i < reverse_i) {
            auto temp = vec[i];
            vec[i] = vec[reverse_i];
            vec[reverse_i] = temp;
        }
    }
}

__global__ void fft_butterflies(size_type *vec, size_type size, size_type len, size_type w_len) {
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < size && i % len == 0; i += blockDim.x * gridDim.x) {
        size_type w = 1;
        auto half_len = len / 2;
        for (auto j = 0; j < half_len; ++j) {
            auto t = size_type(vec[i + j + half_len] * 1ul * w % MOD);
            auto u = vec[i + j];
            vec[i + j] = (u + t) % MOD;
            vec[i + j + half_len] = u >= t ? u - t : u - t + MOD;
            w = w * 1ul * w_len % MOD;
        }
    }
}

__global__ void invert_fft_result(size_type *vec, size_type size, size_type revSize) {
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        vec[i] = vec[i] * 1ul * revSize % MOD;
}

void parallel_fft(size_type *vec, size_type size, bool invert, int numBlocks, int blockSize) {
    cudaError_t err = cudaSuccess;
    bit_reversal<<<numBlocks, blockSize>>>(vec, size, log2(size));
    cudaDeviceSynchronize();
    check_error(err);

    for (auto len = 2; len <= size; len <<= 1) {
        size_type w_len = invert ? reverse(ROOT, MOD) : ROOT;
        for (auto i = len; i < ROOT_ORDER; i <<= 1)
            w_len = size_type(w_len * 1ul * w_len % MOD);
        fft_butterflies<<<numBlocks, blockSize>>>(vec, size, len, w_len);
        cudaDeviceSynchronize();
        check_error(err);
    }

    if (invert) {
        size_type revSize = reverse(size, MOD);
        invert_fft_result<<<numBlocks, blockSize>>>(vec, size, revSize);
        cudaDeviceSynchronize();
        check_error(err);
    }
}

__global__ void multiply_vectors(size_type *vec1, size_type *vec2, size_type *res_vec, size_type size) {
    for (size_type i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        res_vec[i] = size_type(vec1[i] * 1ul * vec2[i] % MOD);
}

int main() {
    cudaError_t err = cudaSuccess;
    int blockSize = 256;
    auto numBlocks = (ROOT_ORDER + blockSize - 1) / blockSize;

    bool random = true;
    std::cout << "random values - 1, hardcoded values - 0: ";
    std::cin >> random;
    std::cout << std::endl;

    size_type *vector1, *vector2;
    cudaMallocManaged(&vector1, ROOT_ORDER * sizeof(size_type));
    cudaMallocManaged(&vector2, ROOT_ORDER * sizeof(size_type));
    if (random) {
        srand(time(NULL));
        for (auto i = 0; i < ROOT_ORDER / 2; i++) {
            vector1[i] = rand() % MOD;
            vector2[i] = rand() % MOD;
        }
    } else {
        std::vector<size_type> test_vec1 = { 41, 6334, 19169, 11478 };
        std::vector<size_type> test_vec2 = { 18467, 26500, 15724, 29358 };
        test_vec1.resize(ROOT_ORDER);
        test_vec2.resize(ROOT_ORDER);
        std::copy(test_vec1.begin(), test_vec1.end(), vector1);
        std::copy(test_vec2.begin(), test_vec2.end(), vector2);
    }

    print_array("vector1", vector1, ROOT_ORDER, true);
    print_array("vector2", vector2, ROOT_ORDER, true);

    parallel_fft(vector1, ROOT_ORDER, false, numBlocks, blockSize);
    parallel_fft(vector2, ROOT_ORDER, false, numBlocks, blockSize);

    print_array("parallel FFT vector1", vector1, ROOT_ORDER, false);
    print_array("parallel FFT vector2", vector2, ROOT_ORDER, false);

    size_type *res_vec;
    cudaMallocManaged(&res_vec, ROOT_ORDER * sizeof(size_type));
    multiply_vectors<<<numBlocks, blockSize>>>(vector1, vector2, res_vec, ROOT_ORDER);
    cudaDeviceSynchronize();
    check_error(err);

    cudaFree(vector1);
    cudaFree(vector2);
    check_error(err);

    print_array("multiplied FFT vector", res_vec, ROOT_ORDER, false);

    parallel_fft(res_vec, ROOT_ORDER, true, numBlocks, blockSize);

    print_array("result", res_vec, ROOT_ORDER, true);

    cudaFree(res_vec);
    check_error(err);
}
