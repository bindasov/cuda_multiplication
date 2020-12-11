#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const unsigned long long MOD = 2543537ul;
const unsigned long long ROOT = 322102ul;
const unsigned long long ROOT_ORDER = 1ul << 4ul;

// бинарное возведение в степень по модулю (a^n(mod))
unsigned long long binary_pow(unsigned long long a, unsigned long long n, unsigned long long mod) {
    unsigned long long res = 1ul;
    while (n) {
        if (n & 1ul) res = res * a % mod;
        a = a * a % mod;
        n >>= 1ul;
    }
    return res;
}

// находит обратный элемент как n^(mod-2)
unsigned long long reverse(unsigned long long n, unsigned long long mod) {
    return binary_pow(n, mod - 2, mod);
}

unsigned long long *generate_roots(unsigned long long *roots_array, unsigned long long root, unsigned long long mod) {
    roots_array[0] = 1ul;
    roots_array[1] = root;
    for (auto i = 2; i < ROOT_ORDER; i++)
        roots_array[i] = roots_array[i - 1] * root % mod;
    return roots_array;
}

__global__ void calc_fft(unsigned long long *fft_vec, unsigned long long *vec, unsigned long long *roots_array) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx < ROOT_ORDER) {
        unsigned long long fft_value = 0ul;
        for (auto j = 0; j < ROOT_ORDER; j++) {
            fft_value += (vec[j] * roots_array[(j * globalIdx) % ROOT_ORDER]);
            fft_value %= MOD;
        }
        fft_vec[globalIdx] = fft_value;
    }
}

__global__ void calc_revert_fft(unsigned long long *vec, unsigned long long *fft_vec, unsigned long long *roots_array,
    unsigned long long reverse) {
    unsigned long long globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx < ROOT_ORDER) {
        unsigned long long fft_value = 0ul;
        for (auto j = 0; j < ROOT_ORDER; j++) {
            fft_value += (fft_vec[j] * roots_array[(j * globalIdx) % ROOT_ORDER]);
            fft_value %= MOD;
        }
        vec[globalIdx] = fft_value * reverse % MOD;
    }
}

__global__ void multiply_vectors(unsigned long long *vec1, unsigned long long *vec2, unsigned long long *res_vec) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx < ROOT_ORDER)
        res_vec[globalIdx] = vec1[globalIdx] * vec2[globalIdx] % MOD;
}

void print_vector(const std::string& name, unsigned long long *vec, bool polynomial) {
    std::cout << name << ": ";
    for (auto i = 0; i < ROOT_ORDER; i++) {
        if (polynomial) {
            if (vec[i] != 0)
                std::cout << vec[i] << "x^" << i << " ";
        }
        else {
            if (i != ROOT_ORDER - 1)
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
        fprintf(stderr, "error code: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    cudaError_t err = cudaSuccess;
    int blockSize = 512;
    unsigned long long numBlocks = (ROOT_ORDER + blockSize - 1) / blockSize;

    bool random = true;
    std::cout << "random values - 1, hardcoded values - 0: ";
    std::cin >> random;
    std::cout << std::endl;

    unsigned long long *vector1, *vector2;
    cudaMallocManaged(&vector1, ROOT_ORDER * sizeof(unsigned long long));
    cudaMallocManaged(&vector2, ROOT_ORDER * sizeof(unsigned long long));
    if (random) {
        for (auto i = 0; i < ROOT_ORDER / 2; i++) {
            vector1[i] = rand() % MOD;
            vector2[i] = rand() % MOD;
        }
    }
    else {
        std::vector<unsigned long long> test_vec1 = { 41, 314, 283, 279 };
        std::vector<unsigned long long> test_vec2 = { 1016, 1605, 1393 };
        test_vec1.resize(ROOT_ORDER);
        test_vec2.resize(ROOT_ORDER);
        std::copy(test_vec1.begin(), test_vec1.end(), vector1);
        std::copy(test_vec2.begin(), test_vec2.end(), vector2);
    }
    print_vector("vector1", vector1, true);
    print_vector("vector2", vector2, true);

    unsigned long long *roots_array;
    cudaMallocManaged(&roots_array, ROOT_ORDER * sizeof(unsigned long long));
    roots_array = generate_roots(roots_array, ROOT, MOD);
    print_vector("roots", roots_array, false);

    unsigned long long *fft_vec1, *fft_vec2;
    cudaMallocManaged(&fft_vec1, ROOT_ORDER * sizeof(unsigned long long));
    cudaMallocManaged(&fft_vec2, ROOT_ORDER * sizeof(unsigned long long));
    calc_fft << < numBlocks, blockSize >> > (fft_vec1, vector1, roots_array);
    cudaDeviceSynchronize();
    check_error(err);
    calc_fft << < numBlocks, blockSize >> > (fft_vec2, vector2, roots_array);
    cudaDeviceSynchronize();
    check_error(err);

    cudaFree(vector1);
    cudaFree(vector2);
    check_error(err);

    print_vector("FFT vector1", fft_vec1, false);
    print_vector("FFT vector2", fft_vec2, false);

    unsigned long long *res_vec, *rev_fft_vec;
    cudaMallocManaged(&res_vec, ROOT_ORDER * sizeof(unsigned long long));
    cudaMallocManaged(&rev_fft_vec, ROOT_ORDER * sizeof(unsigned long long));
    multiply_vectors << < numBlocks, blockSize >> > (fft_vec1, fft_vec2, res_vec);
    cudaDeviceSynchronize();
    check_error(err);

    cudaFree(fft_vec1);
    cudaFree(fft_vec2);
    check_error(err);

    print_vector("multiplied FFT vector", res_vec, false);

    for (auto i = 0; i < ROOT_ORDER; i++)
        roots_array[i] = reverse(roots_array[i], MOD);

    calc_revert_fft << < numBlocks, blockSize >> > (rev_fft_vec, res_vec, roots_array, reverse(ROOT_ORDER, MOD));
    cudaDeviceSynchronize();
    check_error(err);
    print_vector("result", rev_fft_vec, true);

    cudaFree(res_vec);
    cudaFree(rev_fft_vec);
    cudaFree(roots_array);
    check_error(err);
}
