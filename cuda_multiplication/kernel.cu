#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int mod = 2543537;
const int root = 322102;
const int root_pw = 1 << 4;

unsigned long long reverse(unsigned long long n, unsigned long long mod) {
    int reverse = 0;
    for (unsigned long long i = 0; i < mod; i++) {
        if ((i * n) % mod == 1) return i;
    }
    return 0;
}

unsigned long long * generate_roots(unsigned long long * roots_array, int root, int mod) {
    roots_array[0] = 1; roots_array[1] = root;
    for (int i = 2; i < root_pw; i++) {
        roots_array[i] = roots_array[i - 1] * root % mod;
    }
    return roots_array;
}

__global__ void calc_fft(unsigned long long* fft_vec, unsigned long long* vec, unsigned long long* roots_array) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx < root_pw) {
        int fft_value = 0;
        for (int j = 0; j < root_pw; j++)
            fft_value += (vec[j] * roots_array[(j * globalIdx) % root_pw]) % mod;
        fft_vec[globalIdx] = fft_value % mod;
    }
}

__global__ void calc_revert_fft(unsigned long long* vec, unsigned long long* fft_vec, unsigned long long* roots_array, unsigned long long reverse) {
    unsigned long long globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx < root_pw) {
        unsigned long long fft_value = 0;
        for (int j = 0; j < root_pw; j++)
            fft_value += (fft_vec[j] * roots_array[(j * globalIdx) % root_pw]) % mod;
        vec[globalIdx] = (fft_value % mod) * reverse % mod;
    }
}

__global__ void multiply_vectors(unsigned long long* vec1, unsigned long long* vec2, unsigned long long* res_vec) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx < root_pw)
        res_vec[globalIdx] = vec1[globalIdx] * vec2[globalIdx] % mod;
}

void print_vector(std::string name, unsigned long long* vec, bool polynomial) {
    std::cout << name << ": ";
    for (unsigned long long i = 0; i < root_pw; i++) {
        if (polynomial) {
            if (vec[i] != 0)
                std::cout << vec[i] << "x^" << i << " ";
        }            
        else {
            if (i != root_pw - 1)
                std::cout << vec[i] << ", ";
            else
                std::cout << vec[i];
        }
    }
    std::cout << std::endl;
}

void check_error(cudaError_t err) {
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "error code: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    cudaError_t err = cudaSuccess;
    int blockSize = 512;
    int numBlocks = (root_pw + blockSize - 1) / blockSize;

    bool random = true;
    std::cout << "random values - 1, hardcoded values - 0: ";
    std::cin >> random;
    std::cout << std::endl;

    unsigned long long* vector1, * vector2;
    cudaMallocManaged(&vector1, root_pw * sizeof(unsigned long long));
    cudaMallocManaged(&vector2, root_pw * sizeof(unsigned long long));
    if (random) {
        for (int i = 0; i < root_pw / 2; i++) {
            vector1[i] = 0 + rand() % mod;
            vector2[i] = 0 + rand() % mod;
        }
    }
    else {
        std::vector<unsigned long long> test_vec1 = { 41, 314, 283, 279 };
        std::vector<unsigned long long> test_vec2 = { 1016, 1605, 1393 };
        for (int i = 0; i < root_pw; i++) {
            if (i < test_vec1.size())
                vector1[i] = test_vec1[i];
            else
                vector1[i] = 0;
            if (i < test_vec2.size())
                vector2[i] = test_vec2[i];
            else
                vector2[i] = 0;
        }
    }
    print_vector("vector1", vector1, true);
    print_vector("vector2", vector2, true);

    unsigned long long * roots_array;
    cudaMallocManaged(&roots_array, root_pw * sizeof(unsigned long long));
    roots_array = generate_roots(roots_array, root, mod);
    print_vector("roots", roots_array, false);

    unsigned long long* fft_vec1, * fft_vec2;
    cudaMallocManaged(&fft_vec1, root_pw * sizeof(unsigned long long));
    cudaMallocManaged(&fft_vec2, root_pw * sizeof(unsigned long long));
    calc_fft <<<numBlocks, blockSize >>> (fft_vec1, vector1, roots_array);
    cudaDeviceSynchronize();
    check_error(err);
    calc_fft <<<numBlocks, blockSize >>> (fft_vec2, vector2, roots_array);
    cudaDeviceSynchronize();
    check_error(err);

    cudaFree(vector1);
    cudaFree(vector2);
    check_error(err);

    print_vector("FFT vector1", fft_vec1, false);
    print_vector("FFT vector2", fft_vec2, false);
 
    unsigned long long* res_vec, * rev_fft_vec;
    cudaMallocManaged(&res_vec, root_pw * sizeof(unsigned long long));
    cudaMallocManaged(&rev_fft_vec, root_pw * sizeof(unsigned long long));
    multiply_vectors <<<numBlocks, blockSize >>> (fft_vec1, fft_vec2, res_vec);
    cudaDeviceSynchronize();
    check_error(err);

    cudaFree(fft_vec1);
    cudaFree(fft_vec2);
    check_error(err);

    print_vector("multiplied FFT vector", res_vec, false);

    for (unsigned long long i = 0; i < root_pw; i++) {
        roots_array[i] = reverse(roots_array[i], mod);
    }

    calc_revert_fft <<<numBlocks, blockSize >>> (rev_fft_vec, res_vec, roots_array, reverse(root_pw, mod));
    cudaDeviceSynchronize();
    check_error(err);
    print_vector("result", rev_fft_vec, true);

    cudaFree(res_vec);
    cudaFree(rev_fft_vec);
    cudaFree(roots_array);
    check_error(err);
};