#include <cuda_runtime.h>
#include <iostream>
// nvcc -o device_query device_query.cu
// ./device_query
using namespace std;
int main()
{
    cudaDeviceProp deviceProp; // Structure to hold device properties
    int dev = 0; // Querying device 0
    cudaGetDeviceProperties(&deviceProp, dev); // Get properties of device 0

    cout << "Device Name: " << dev << ": " << deviceProp.name << endl; // ASCII string identifying device

    cout << " CUDA Capability Major/Minor version number: " <<
        deviceProp.major << "." << deviceProp.minor << endl; // Major and minor compute capability

    cout << " Total amount of shared memory per block: " <<
        deviceProp.sharedMemPerBlock << " bytes" << endl; // Shared memory available per block in bytes

    cout << " Maximum number of threads per block: " <<
        deviceProp.maxThreadsPerBlock << endl; // Maximum number of threads per block

    return 0;
}