#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;
//nvcc -o vector_scalar_multiplication vector_scalar_multiplication.cu
//./vector_scalar_multiplication

__global__ void vectorScalarMultiplicationKernel(float *A, float *C, int n, float scalar)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        C[i] = A[i] * scalar;
    }
}

__global__ void vectorOperationKernel(float *A, float *B, float *C, int n, string operation)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (operation == "multiplication")
    {
        C[i] = A[i] * B[i];
    }
    else if (operation == "addition")
    {
        C[i] = A[i] + B[i];
    }
    else if (operation == "subtraction")
    {
        C[i] = A[i] - B[i];
    }
    else if (operation == "absolute_difference")
    {
        C[i] = abs(A[i] - B[i]);
    }
    else if (operation == "maximum")
    {
        C[i] = max(A[i], B[i]);
    }
    else if (operation == "minimum")
    {
        C[i] = min(A[i], B[i]);
    }
    else if (operation == "modulus")
    {
        C[i] = fmod(A[i], B[i]);
    }
    else if (operation == "division")
    {
        C[i] = A[i] / B[i];
    }
    else if (operation == "power")
    {
        C[i] = pow(A[i], 2);
    }
    else if (operation == "square_root")
    {
        C[i] = sqrt(A[i]);
    }
    else if (operation == "logarithm")
    {
        C[i] = log(A[i]);
    }
    else if (operation == "exponentiation")
    {
        C[i] = exp(A[i]);
    }

}


int main()
{
    int N = 100'000'000; // Number of elements in the vectors
    size_t size = N * sizeof(float); // Size of the vectors in bytes

    float *h_A = (float *)malloc(size); // Allocate memory for host arrays
    float *h_C = (float *)malloc(size);

    float *d_A; // Allocate memory for device arrays
    float *d_C;
    cudaMalloc((void **)&d_A, size); 
    cudaMalloc((void **)&d_C, size);

    int threadsPerBlock = 1024; // Number of threads per block
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Number of blocks in the grid

    vectorScalarMultiplicationKernel << <blocksPerGrid, threadsPerBlock>> > (d_A, d_C, N, 2.0f); // Launch kernel

    // Copy results back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);
    return 0;

}