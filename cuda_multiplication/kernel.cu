#include <cstdlib>
#include <vector>
#include <iostream>
#include <string>
#include <cmath>

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

void print_vector(const std::string &name, const size_type *vec, size_type size, bool polynomial) {
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

void fft(std::vector<size_type> &a, bool invert) {
    int n = (int)a.size();

    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1)
            j -= bit;
        j += bit;
        if (i < j)
            std::swap(a[i], a[j]);
    }
    //print_vector("bit_reversal vector", a.data(), n, false);

    for (int len = 2; len <= n; len <<= 1) {
        int wlen = invert ? reverse(ROOT, MOD) : ROOT;
        for (int i = len; i < n; i <<= 1)
            wlen = int(wlen * 1ll * wlen % MOD);
        //std::cout << "len = " << len << " wlen = " << wlen << std::endl;
        for (int i = 0; i < n; i += len) {
            int w = 1;
            for (int j = 0; j < len / 2; ++j) {
                int u = a[i + j], v = int(a[i + j + len / 2] * 1ll * w % MOD);
                a[i + j] = u + v < MOD ? u + v : u + v - MOD;
                a[i + j + len / 2] = u - v >= 0 ? u - v : u - v + MOD;
                w = int(w * 1ll * wlen % MOD);
            }
        }
        // print_vector("fft_butterflies len " + std::to_string(len), a.data(), n, false);
    }

    if (invert) {
        int nrev = reverse(n, MOD);
        for (int i = 0; i < n; ++i)
            a[i] = int(a[i] * 1ll * nrev % MOD);
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

__global__ void fft_butterflies(size_type *vec, size_type size, size_type len, size_type wlen) {
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < size && i % len == 0; i += blockDim.x * gridDim.x) {
        size_type w = 1;
        auto half_len = len / 2;
        for (auto j = 0; j < half_len; ++j) {
            auto t = size_type(vec[i + j + half_len] * 1ul * w % MOD);
            auto u = vec[i + j];
            vec[i + j] = (u + t) % MOD;
            vec[i + j + half_len] = u >= t ? u - t : u - t + MOD;
            w = w * 1ul * wlen % MOD;
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
    //print_vector("parallel bit_reversal vector", vec, size, false);

    for (auto len = 2; len <= size; len <<= 1) {
        size_type wlen = invert ? reverse(ROOT, MOD) : ROOT;
        for (auto i = len; i < ROOT_ORDER; i <<= 1)
            wlen = size_type(wlen * 1ul * wlen % MOD);
        //std::cout << "parallel len = " << len << " wlen = " << wlen << std::endl;

        fft_butterflies<<<numBlocks, blockSize>>>(vec, size, len, wlen);
        //cudaDeviceSynchronize();
        //check_error(err);
        //print_vector("parallel fft_butterflies len " + std::to_string(len), vec, size, false);
    }

    cudaDeviceSynchronize();
    check_error(err);

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
        /*fft(test_vec1, false);
        print_vector("fft vector1", test_vec1.data(), ROOT_ORDER, false);
        fft(test_vec2, false);
        print_vector("fft vector2", test_vec2.data(), ROOT_ORDER, false);*/
    }

    print_vector("vector1", vector1, ROOT_ORDER, true);
    print_vector("vector2", vector2, ROOT_ORDER, true);

    parallel_fft(vector1, ROOT_ORDER, false, numBlocks, blockSize);
    parallel_fft(vector2, ROOT_ORDER, false, numBlocks, blockSize);

    print_vector("parallel FFT vector1", vector1, ROOT_ORDER, false);
    print_vector("parallel FFT vector2", vector2, ROOT_ORDER, false);

    size_type *res_vec;
    cudaMallocManaged(&res_vec, ROOT_ORDER * sizeof(size_type));
    multiply_vectors<<<numBlocks, blockSize>>>(vector1, vector2, res_vec, ROOT_ORDER);
    cudaDeviceSynchronize();
    check_error(err);

    cudaFree(vector1);
    cudaFree(vector2);
    check_error(err);

    print_vector("multiplied FFT vector", res_vec, ROOT_ORDER, false);

    parallel_fft(res_vec, ROOT_ORDER, true, numBlocks, blockSize);

    print_vector("result", res_vec, ROOT_ORDER, true);

    cudaFree(res_vec);
    check_error(err);
}
