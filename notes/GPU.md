## Threads
1. Basic unit of execution in CUDA
2. Each thread has its own id so it can access its own part of the data 
3. Every thread has local memory and registers (registers store local varabiables such as tid)
4. Each thread has a finite amount of registers so if we define too many local variables that can cause
register spillage

## Warps
1. Groups of 32 threads
2. All threads in a warp execute the same instruction all at the same time

## Blocks 
1. Groups of threads that execute a kernal together
2. Containers for threads
3. Number of threads per block is block size 

## Grid
1. Collection of blocks used to execute a kernel
2. Number of blocks in a grid is grid size 

## Stream
1. A queue of operations that execute in order on the GPU
2. Operations on the same stream run sequentially
3. Operations on different streams run concurrenlty

## Event
1. Used to track GPU execution time

## Amdahls Law
1.  The speed up we achieve with we can achieve with parralalism is by the parts of our application that are sequential

## Cache Miss 
1. When the CPU or GPU looks for data in the cache but nothing is there so it looks for the data somewhere else and that wait is called the miss