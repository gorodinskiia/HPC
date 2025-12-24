#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;
//nvcc -o vectorAdd vectorAdd.cu
//./vectorAdd

//The arrays here live in GPU memeory so we need to pass in pointers
//We always pass in pointers to CUDA kernels because the GPU can only access memory that is in its own memory space
//We use the __global__ keyword to declare a CUDA kernel
__global__ void vectorAddKernel(float *A, float *B, float *C, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void vectorScalarMultiplicationKernel(float *A, float *C, int n, float scalar)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        C[i] = A[i] * scalar;
    }
}

void vectorAddCpu(float *A, float *B, float *C, int n)
{
    for(int i = 0; i<n; i++)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    //float *h_A = (float *)malloc(1000);
    //为h_A分配1000个float类型的内存空间 bro what the fuck is that you mean by that
    //1. float *h_A is a pointer to a float type variable
    //2. h means that it is a host variable, which means it is stored in the CPU memory
    //3. h_A → [ float ][ float ][ float ] ... [ float ] pointing to the first float in the array
    //4. malloc(1000) is a function that allocates 1000 bytes of memory in the heap
    //5. malloc function originally returns a void pointer, so we need to cast it to a float pointer
    //6. Memory is uninitialized, so the values in the memory are undefined
    //7. malloc only works with bytes, so 1000 bytes of memory is allocated
    //8. so, h_A is a pointer to a float type variable, and it points to the first float in the array of 1000 bytes of memory allocated by malloc

    //float *d_A;
    //cudaMalloc((void **)&d_A, 1000);
    //为d_A分配1000个float类型的内存空间
    //1. cudaMalloc is a function that allocates memory on the GPU
    //2. (void **)&d_A is a pointer to a pointer to a void type variable
    //3. d_A is a pointer to a float type variable, and it points to the first 
    //float in the array of 1000 bytes of memory allocated by cudaMalloc
    //4. 1000 is the size of the memory to be allocated in bytes
    //5. cudaMalloc returns 0 if successful, and a non-zero value if it fails
    //6. (void **)&d_A is a pointer to a pointer to a void type variable, so we need to cast it to a pointer to a float type variable
    //7. so, d_A is a pointer to a float type variable, and it points to the first float in the array of 1000 bytes of memory allocated by cudaMalloc

    int N = 100'000'000; // Number of elements in the vectors
    size_t size = N * sizeof(float); // Size of the vectors in bytes

    float *h_A = (float *)malloc(size); // Allocate memory for host arrays
    float *h_B = (float *)malloc(size); 
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N; ++i) { 
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    } // Initialize host arrays

    float *d_A; // Allocate memory for device arrays
    float *d_B;
    float *d_C;
    cudaMalloc((void **)&d_A, size); 
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    int threadsPerBlock = 1024; // Number of threads per block
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Number of blocks in the grid

    cudaEvent_t startEvent, stopEvent; // Create events to measure time
    cudaEventCreate(&startEvent); // Create events to measure time
    cudaEventCreate(&stopEvent); // Create events to measure time

    cudaEventRecord(startEvent, 0); // Record start event
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); // Copy data from host to device
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); // Copy data from host to device

    cudaEventRecord(stopEvent, 0); // Record stop event
    cudaEventSynchronize(stopEvent); // Wait for the stop event to be recorded

    float gpuCopyTime = 0; // Time taken to copy data to GPU
    cudaEventElapsedTime(&gpuCopyTime, startEvent, stopEvent); // Calculate time taken to copy data to GPU

    std::cout<< std::fixed << "Time to copy data to GPU: " << gpuCopyTime << " ms" << std::endl; // Print time taken to copy data to GPU

    cudaEventRecord(startEvent, 0); // Record start event

    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); // Launch kernel

    cudaEventRecord(stopEvent, 0); // Record stop event
    cudaEventSynchronize(stopEvent); // Wait for the stop event to be recorded

    float gpuExecutionTime = 0; // Time taken to execute kernel
    cudaEventElapsedTime(&gpuExecutionTime, startEvent, stopEvent);     // Calculate time taken to execute kernel

    std::cout<< std::fixed << "Time to execute on GPU: " << gpuExecutionTime << " ms" << std::endl; // Print time taken to execute kernel

    cudaEventRecord(startEvent, 0); // Record start event

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); // Copy results back to host

    cudaEventRecord(stopEvent, 0); // Record stop event
    cudaEventSynchronize(stopEvent); // Wait for the stop event to be recorded

    float gpuRetrieveTime = 0; // Time taken to copy results back to host
    cudaEventElapsedTime(&gpuRetrieveTime, startEvent, stopEvent); // Calculate time taken to copy results back to host

    std::cout<< std::fixed << "Time taken to copy results back GPU: " << gpuRetrieveTime << " ms" << std::endl << std::endl; // Print time taken to copy results back to host

    float gpuDuration = (gpuCopyTime + gpuExecutionTime + gpuRetrieveTime); // Total time taken by GPU
    std::cout << "Time taken by GPU: " << gpuDuration << " ms" << std::endl; // Print total time taken by GPU


    cudaEventDestroy(startEvent); // Destroy events
    cudaEventDestroy(stopEvent); // Destroy events


    auto start = std::chrono::high_resolution_clock::now(); // Start timer

    vectorAddCpu(h_A, h_B, h_C, N); // Execute CPU function

    auto stop = std::chrono::high_resolution_clock::now(); // Stop timer
    std::chrono::duration<double, std::milli> cpuDuration = (stop - start); // Calculate time taken by CPU

    std::cout << "Time taken by CPU: " << cpuDuration.count() << " ms" << std::endl; // Print time taken by CPU
    std::cout << "========================================== " << std::endl;  // Print separator

    std::cout << "speed up (execution time only): " << cpuDuration.count() / gpuExecutionTime << std::endl; // Print speed up (execution time only)
    std::cout << "speed up (GPU total time): " << cpuDuration.count() / gpuDuration << std::endl; // Print speed up (GPU total time)


    cudaFree(d_A); // Free device memory
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);



    return 0; // Return 0 to indicate successful execution

}