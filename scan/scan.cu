#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

/**
 * Collect last element from each block for block-level scan
 * Execution pattern:
 * - One thread block per input block
 * - Only thread 255 (last thread) in each block writes to blockSums
 * - Total writes to blockSums = numBlocks
 */
__global__ void collect_block_sums(int* output, int N, int* blockSums)
{
    // get the block id and thread id since each THREAD will be executing this function
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;

    // this is the LAST thread in the block (thread 255)
    if (threadId == THREADS_PER_BLOCK - 1) {
        // then set the blockSum for this block to the that that threads corresponding location in the array
        blockSums[blockId] = output[blockId * THREADS_PER_BLOCK + threadId];
    }
}

/**
 * Add block sums back to all elements in each block
 * Execution pattern:
 * - One thread per element in input array
 * - All threads except those in first block add their block's scanned sum
 * - Total updates = N - THREADS_PER_BLOCK (all elements except first block)
 */
__global__ void add_block_sums(int* output, int N, int* blockSums)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    // add the scanned sum of all prev blocks
    if (blockIdx.x > 0) {
        output[gid] += blockSums[blockIdx.x - 1];
    }
}

/**
 * Perform scan within a single warp (32 threads)
 * Execution pattern:
 * - Called by each thread within a warp
 * - idx is the local index within the warp (0-31)
 * - Each thread updates its own value based on previous values
 */
__device__ int scan_warp(int* ptr, const unsigned int idx)
{
    // Work: O(nlogn), Span: 5 (log base 2 32)
    // idx is the CUDA thread index of the caller: question: @AI: does this thread number reset for every block?
    // TODO: Think about how to handle the doubling pattern (@AI: please flesh out this question more)
    // "hey, every cuda thread other than first one, pls add up your value and the one to the left"
    const unsigned int lane = idx % 32;

    if (lane >= 1) ptr[idx] = ptr[idx - 1] + ptr[idx];
    // "if i'm a thread higher than lane 2, I'll add my predecessor's sum that includes 2 elements"
    if (lane >= 2) ptr[idx] = ptr[idx - 2] + ptr[idx];
    // "if i'm a thread higher than lane 4, I'll add my predecessor's sum that includes 4 elements"
    if (lane >= 4) ptr[idx] = ptr[idx - 4] + ptr[idx];
    // etc
    if (lane >= 8) ptr[idx] = ptr[idx - 8] + ptr[idx];
    if (lane >= 16) ptr[idx] = ptr[idx - 16] + ptr[idx];

    return (lane > 0 ) ptr[idx - 1] : 0;
}

/**
 * Perform scan within a block (256 threads)
 * Execution pattern:
 * - One thread block (256 threads) processes THREADS_PER_BLOCK elements
 * - Organized into 8 warps of 32 threads each
 * - Three phase process: warp scan → warp sum scan → add back
 */
__global__ void scan_block(int* output, int* input, int* blockSums, int N)
{
    // Think about:
    // - Using shared memory for the block's data
    // - Coordinating warps within the block
    // - Handling the upsweep and downsweep phases
    // we want calc the sums for a block. so we have each block running this function and each 
    // block has 256 threads and for sets of 32 consecutive threads we want to execute them in parallel in a SIMD style instruction block
    // and so we want to call scan_warp for each set of 32
    // scan warp takes in a ptr to the start of the array and assumes 32 elements to work on 
    // @AI please correct and improve the above phrasing. help me think through how exactly to string together this function with the scan_warp function

    // @AI where do we get the pointer and thread index to pass to scan_warp?

    // shared memory for the block (256 wide array)
    __shared__ int temp[THREADS_PER_BLOCK];
    __shared__ int warpSums[8];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;

    // all of the threads are loading an element into the block from the input array
    temp[tid] = (gid < N) ? input[gid] : 0;
    // add a syncthreads to make sure no threads start the warp operation until we've finished loading the block's shared memory
    __syncthreads();

    // alright so now we have our chunk of 256 elements ready to be scanned in batches of 32

    int warpId = tid / 32; // which warp am i in? (0-7 since we have 256/32 = 8 warps)
    int laneId = tid % 32; // which lane am i in? (0-31 since we have 32 wide SIMD)

    scan_warp(&temp[warpId * 32], tid);
    __syncthreads();

    // 2. extract the sums from last lanes of warps and write it to warpSums
    if (laneId == 31) {
        warpSums[warpId] = temp[tid];
    }
    __syncthreads();

    // TODO: understand this code block better
    if (warpId == 0 && localWarpIndex < 8) {
        int t = warpSums[localWarpIndex];
        // Manual scan of just 8 elements
        if (localWarpIndex >= 1) t += warpSums[localWarpIndex - 1];
        if (localWarpIndex >= 2) t += warpSums[localWarpIndex - 2];
        if (localWarpIndex >= 4) t += warpSums[localWarpIndex - 4];
        warpSums[localWarpIndex] = t;
    }
    __syncthreads();

    // add the warp sums back 
    if (warpId > 0) {
        temp[tid] += warpSums[warpId - 1]
    }
    __syncthreads();

    if (tid % 32 == 0) {
        int warpSum = temp[tid + 31];
        __syncthreads();
        if (warpId > 0) {
            temp[tid + 31] += temp[warpOffset - 1];
        }
        __syncthreads();
    }

    if (warpId > 0 && tid > 0) {
        temp[tid] += temp[warpOffset - 1];
    }
    __syncthreads();

    if (gid < N) {
        output[gid] = temp[tid];
    }
    
    // Save block sum for next level
    if (tid == THREADS_PER_BLOCK - 1) {
        blockSums[bid] = temp[tid];
    }
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result

/**
 * Host function to coordinate the scan operation
 * Execution organization:
 * - Breaks input into blocks of 256 elements
 * - Each block processed by one CUDA thread block
 * - Recursive for large arrays (multiple levels of block sums)
 */
void exclusive_scan(int *input, int N, int *result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.

    // 1. calc grid and block dims
    const int blockSize = THREADS_PER_BLOCK;
    const int numBlocks = (N + blockSize - 1) / blockSize; // think of this on the order of 10M/256 ~ 40K

    // 2. we'll need to store numBlocks block sums in shared block memory
    int* blockSums;
    cudaMalloc(&blockSums, sizeof(int) * numBlocks);

    // 3. scan within blocks
    scan_block<<<numBlocks, blockSize>>>(result, input, blockSums, N)
    cudaDeviceSynchronize();

    // 4. combine sums - TODO LATER: see if recursing all the way down to 1 incurs too much overhead.
    if (numBlocks > 1) {
        // a. fill in blockSums: collect last elements from each block for block-level scan
        collect_block_sums<<<numBlocks, blockSize>>>(result, N, blockSums);
        cudaDeviceSynchronize();

        // b. scan the block sums
        int* scanned_block_sums;
        // we're allocating a new result array on GPU. on the first recusive call, the INPUT and OUTPUT are now both numBlock big arrays instead of N big arrays
        cudaMalloc(&scanned_block_sums, sizeof(int) * numBlocks);
        // run the scan, filling in scanned_block_sums
        exclusive_scan(blockSums, numBlocks, scanned_block_sums);

        // c. add block sums back to elements
        add_block_sums<<<numBlocks, blockSize>>>(result, N, scanned_block_sums);
        cudaDeviceSynchronize();

        cudaFree(scanned_block_sums);
    }

    cudaFree(blockSums);
}

//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int *inarray, int *end, int *resultarray)
{
    int *device_result;
    int *device_input;
    int N = end - inarray;

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);

    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration;
}

// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int *inarray, int *end, int *resultarray)
{

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration;
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int *device_input, int length, int *device_output)
{

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    return 0;
}

//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length)
{

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);

    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();

    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime;
    return duration;
}

void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
