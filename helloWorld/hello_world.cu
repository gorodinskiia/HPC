#include <cuda_runtime.h>
#include <iostream>
// nvcc -o hello_world hello_world.cu
// ./hello_world
__global__ void hello_world()
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello World from thread %d\n", tid);
}

int main()
{
    hello_world<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}