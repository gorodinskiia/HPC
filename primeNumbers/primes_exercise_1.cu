#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
using namespace std;
// nvcc -o primes_exe_1 primes_exercise_1.cu
// ./primes_exe_1

__global__ void checkPrimeKernel(long long start, long long end, long long *d_numbersTested_array, long long *d_primes_array) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    long long num = start + (tid * 2);
    bool isPrime = true;

    if (num <= 1) {
        isPrime = false;
        d_numbersTested_array[tid] = num;
        d_primes_array[tid] = isPrime;
        return;
    }
    if (num == 2) {
        isPrime = true;
        d_numbersTested_array[tid] = num;
        d_primes_array[tid] = isPrime;
        return;
    } 
    if (num % 2 == 0) {
        isPrime = false;
        d_numbersTested_array[tid] = num;
        d_primes_array[tid] = isPrime;
        return;
    }
    if (num > end)
    {
        d_numbersTested_array[tid] = num;
        d_primes_array[tid] = isPrime;
        return;
    }
    for (long long i = 3; i * i <= num; i += 2) {

        if (num % i == 0) {
            isPrime = false;
            break;
        }

    }

    d_numbersTested_array[tid] = num;
    d_primes_array[tid] = isPrime;

    /*
    * for study purposes we can print the verification of each number
    */
    if (tid < 10) 
    {  // only print first 10 threads
        printf("tid=%lld num=%lld isPrime=%d storedInNumbersTested=%lld storedInPrimes=%d\n",
           (long long)tid,
           num,
           isPrime,
           d_numbersTested_array[tid],
           d_primes_array[tid]);
    }

}

bool checkPrimeCpu(long long num) {
    
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (long long i = 3; i * i <= num; i += 2) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}


int main() {
    long long start =  100'001LL; // must start with odd
    long long end   =  190'001LL; // must end with odd

    int threadsPerBlock = 256;
    int totalNumbers = (end - start) / 2 + 1;
    int blocksPerGrid = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;

    long long *d_numbersTested_array;
    long long *d_primes_array;
    cudaMalloc(&d_numbersTested_array, totalNumbers * sizeof(long long));
    cudaMalloc(&d_primes_array, totalNumbers * sizeof(long long));



    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    checkPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(start, end, d_numbersTested_array, d_primes_array);
    cudaDeviceSynchronize();

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);

    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_numbersTested_array);
    cudaFree(d_primes_array);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (long long num = start; num <= end; num += 2) {
        checkPrimeCpu(num);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endTime - startTime;

    std::cout << "Time taken on CPU: " << std::fixed << cpuDuration.count() << " ms" << std::endl;
    std::cout << "speed up : " << cpuDuration.count() / gpuDuration << std::endl;

    return 0;
}