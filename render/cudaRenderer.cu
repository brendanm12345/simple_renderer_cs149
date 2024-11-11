#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"
#define TILE_SIZE 64
#define SCAN_BLOCK_DIM 1024
#define RENDER_CHUNK_SIZE 512
#include "exclusiveScan.cu_inl"
#include "circleBoxTest.cu_inl"

#include <thrust/scan.h> 

////////////////////////////////////////////////////////////////////////////////////////
// CUDA Error Checking
///////////////////////////////////////////////////////////////////////////////////////

#define DEBUG  // comment this out when we're done debugging

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", 
              cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif

// Also useful to catch errors from kernel launches
inline void checkCudaErrors(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Cuda Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    
    // Compute gradient based on y position
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    
    // Debug first few pixels
    if (imageY < 2 && imageX == 0) {
        printf("Setting background at [%d][%d] to %f\n", 
               imageX, imageY, shade);
    }

    float4 value = make_float4(shade, shade, shade, 1.f);
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {
    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];
    float maxDist = rad * rad;

    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    if (cuConstRendererParams.sceneName == SNOWFLAKES || 
        cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
    } else {
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // Read current color, ensuring we have valid initial values
    float4 existingColor = *imagePtr;
    if (isnan(existingColor.x) || isnan(existingColor.y) || isnan(existingColor.z)) {
        existingColor = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
    }

    // Compute new color
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // Debug output for pixel [0,0]
    if (pixelCenter.x == 0.5f/cuConstRendererParams.imageWidth && 
        pixelCenter.y == 0.5f/cuConstRendererParams.imageHeight) {
        // printf("Circle %d: rgb=(%f,%f,%f) alpha=%f\n", 
        //        circleIndex, rgb.x, rgb.y, rgb.z, alpha);
        // printf("Existing: (%f,%f,%f,%f) -> New: (%f,%f,%f,%f)\n",
        //        existingColor.x, existingColor.y, existingColor.z, existingColor.w,
        //        newColor.x, newColor.y, newColor.z, newColor.w);
    }

    *imagePtr = newColor;
}

// phase 1a: Just compute intersections
__global__ void kernelComputeIntersections(
    int numCircles,
    int numTilesX,
    int numTilesY,
    char* intersectionMatrix
) {
    // One thread per circle
    int circleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (circleIndex >= numCircles) return;

    int index3 = 3 * circleIndex;
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float rad = cuConstRendererParams.radius[circleIndex];

    // early rejection
    if (rad < 0.0001f || // too small
        p.x + rad < 0.0f || p.x - rad > 1.0f ||  // outside x bounds
        p.y + rad < 0.0f || p.y - rad > 1.0f) {  // outside y bounds
        return;
    }

    // Handle snowflake wrapping
    if (cuConstRendererParams.sceneName == SNOWFLAKES ||
        cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
        while (p.y - rad > 1.0f) p.y -= 1.0f;
        while (p.y + rad < 0.0f) p.y += 1.0f;
    }

    // Compute bounding box and tile range
    float boxL = p.x - rad;
    float boxR = p.x + rad;
    float boxB = p.y - rad;
    float boxT = p.y + rad;

    int minTileX = max(0, static_cast<int>(floor(boxL * numTilesX)));
    int maxTileX = min(numTilesX - 1, static_cast<int>(ceil(boxR * numTilesX)));
    int minTileY = max(0, static_cast<int>(floor(boxB * numTilesY)));
    int maxTileY = min(numTilesY - 1, static_cast<int>(ceil(boxT * numTilesY)));

    // For each potentially intersecting tile
    for (int tileY = minTileY; tileY <= maxTileY; tileY++) {
        for (int tileX = minTileX; tileX <= maxTileX; tileX++) {            
            int tileIndex = tileY * numTilesX + tileX;
            
            float tileBoundsL = static_cast<float>(tileX) / numTilesX;
            float tileBoundsR = static_cast<float>(tileX + 1) / numTilesX;
            float tileBoundsB = static_cast<float>(tileY) / numTilesY;
            float tileBoundsT = static_cast<float>(tileY + 1) / numTilesY;
            
            if (circleInBoxConservative(p.x, p.y, rad, 
                                      tileBoundsL, tileBoundsR, tileBoundsT, tileBoundsB)) {
                if (circleInBox(p.x, p.y, rad, 
                              tileBoundsL, tileBoundsR, tileBoundsT, tileBoundsB)) {
                    // printf("Circle %d affects tile [%d,%d] (global index %d)\n", 
                    //     circleIndex, tileX, tileY, tileIndex);
                    intersectionMatrix[tileIndex * numCircles + circleIndex] = 1;
                }
            }
        }
    }
}

// phase 1b: compute tile counts using warp-level operations
__global__ void kernelComputeTileCounts(
    int numCircles,
    int numTilesX,
    int numTilesY,
    char* intersectionMatrix,
    int* tileCounts
) {
    int tileIndex = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (tileIndex >= numTilesX * numTilesY) return;

    int laneId = threadIdx.x & 31;
    
    int matrixRowStart = tileIndex * numCircles;
    
    int total = 0;
    
    
    for (int circleStart = 0; circleStart < numCircles; circleStart += 32) {
        int circleIdx = circleStart + laneId;
        bool hasIntersection = (circleIdx < numCircles) && 
                              intersectionMatrix[matrixRowStart + circleIdx];
        
        unsigned int intersectionMask = __ballot_sync(0xffffffff, hasIntersection);
        
        if (laneId == 0) {
            total += __popc(intersectionMask);
        }
    }
    
    if (laneId == 0) {
        tileCounts[tileIndex] = total;
    }
}

// handle prefix sum of tile counts (small array doesn't need to be parallelized)
__global__ void kernelComputeOffsets(int* tileCounts, int* tileOffsets, int numTiles) {
    // might want to parallelize this for very large numTiles
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int runningSum = 0;
        tileOffsets[0] = 0;  // first offset is always 0
        
        for (int i = 0; i < numTiles; i++) {
            int currentCount = tileCounts[i];
            tileOffsets[i + 1] = runningSum + currentCount;
            runningSum += currentCount;
        }
    }
}

// pack tile info together to improve memory access
struct TileInfo {
    int tileIndex;      // which tile we're processing
    int baseOffset;     // where this tile starts writing in the output array
    int numCircles;     // how many circles intersect this tile
};

__global__ void kernelBuildTileLists(
    int numCircles,
    char* intersectionMatrix,   // [numTiles][numCircles] matrix
    int* tileCounts,            // [numTiles] array of circle counts per tile
    int* tileOffsets,           // [numTiles+1] array of output offsets
    int* tileCircleLists,       // output array containing circle indices
    int numTilesX,
    int numTilesY
) {
    int tileIndex = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (tileIndex >= numTilesX * numTilesY) return;

    // get lane ID within warp (0-31)
    int laneId = threadIdx.x & 31;
    
    
    // in our old implementation we might access: [0-255], [256-511], ...
    // our new way accesses: [0-31], [32-63], ... (perfect warp alignment)
    int matrixRowStart = tileIndex * numCircles;
    int baseOffset = tileOffsets[tileIndex];
    int numCirclesInTile = tileCounts[tileIndex];
    
    // process circles in chunks of 32 to ensure optimal memory coalescing
    for (int circleStart = 0; circleStart < numCircles; circleStart += 32) {
        // each thread in warp loads one circle's intersection data
        int circleIdx = circleStart + laneId;
        bool hasIntersection = (circleIdx < numCircles) && 
                              intersectionMatrix[matrixRowStart + circleIdx];
         
        // create a MASK of which threads have intersections
        unsigned int intersectionMask = __ballot_sync(0xffffffff, hasIntersection);
        
        // popc is like exclusive scan but for binary arrays. so this counts the number of 1s in intersectionMask
        int numIntersectionsBefore = __popc(intersectionMask & ((1u << laneId) - 1));
        
        // if intersects, write circleId to global array (writes are coalesced since positions are consecutive) 
        if (hasIntersection) {
            tileCircleLists[baseOffset + numIntersectionsBefore] = circleIdx;
        }
        
        // update base offset for next chunk by adding total number of intersections found in this warp
        baseOffset += __popc(intersectionMask);
    }
}

// // pack the data for a circle next to each other for quick access
// struct CircleData {
//     float3 position;
//     float radius;
// };

struct CircleData {
    float3 position;
    float radius;
    float3 color;
    float alpha;
};

__global__ void kernelRenderTiles(
    int numCircles,
    int imageWidth,
    int imageHeight,
    int* tileOffsets,
    int* tileCircleLists,
    int numTilesX,
    int numTilesY
) { 
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    if (tileX >= numTilesX || tileY >= numTilesY) return;

    int tileIndex = tileY * numTilesX + tileX;
    
    // calculate tile boundaries
    int tileStartX = tileX * TILE_SIZE;
    int tileStartY = tileY * TILE_SIZE;
    int tileEndX = min(tileStartX + TILE_SIZE, imageWidth);
    int tileEndY = min(tileStartY + TILE_SIZE, imageHeight);
    
    // allocate shared memory arrays visible to all threads in the block
    __shared__ CircleData sharedCircles[RENDER_CHUNK_SIZE];
    __shared__ float3 sharedColors[RENDER_CHUNK_SIZE];
    
    // calculate the number of circles that overlap this tile
    int startOffset = tileOffsets[tileIndex];
    int endOffset = tileOffsets[tileIndex + 1];
    int numCirclesInTile = endOffset - startOffset;
    
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // process circles in chunks, each iteration handles 256 circles
    for (int chunkStart = 0; chunkStart < numCirclesInTile; chunkStart += RENDER_CHUNK_SIZE) {
        int circlesInChunk = min(RENDER_CHUNK_SIZE, numCirclesInTile - chunkStart);
        
        // load circle data into shared memory (use a strided pattern to interleave loading)
        for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < circlesInChunk; i += blockDim.x * blockDim.y) {
            int circleIndex = tileCircleLists[startOffset + chunkStart + i];
            int index3 = 3 * circleIndex;
            
            // load position and radius
            sharedCircles[i].position = *(float3*)(&cuConstRendererParams.position[index3]);
            sharedCircles[i].radius = cuConstRendererParams.radius[circleIndex];
            
            // load color
            sharedColors[i] = *(float3*)(&cuConstRendererParams.color[index3]);
        }
        __syncthreads();
        
        // each thread processes multiple pixels, also in a strided manner
        for (int py = threadIdx.y; py < TILE_SIZE; py += blockDim.y) {
            int pixelY = tileStartY + py;
            if (pixelY >= imageHeight) continue;
            
            for (int px = threadIdx.x; px < TILE_SIZE; px += blockDim.x) {
                int pixelX = tileStartX + px;
                if (pixelX >= imageWidth) continue;

                // calculate global memory location for this pixel
                float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
                // calculate pixel coordinates to normalized space [0,1]
                float2 pixelCenterNorm = make_float2(
                    invWidth * (static_cast<float>(pixelX) + 0.5f),
                    invHeight * (static_cast<float>(pixelY) + 0.5f));

                // key: use local var to avoid repeated global memory access
                float4 accumColor = *imgPtr;
                
                // loop through our chunk of circles and shade the pixels they contribute to
                for (int i = 0; i < circlesInChunk; i++) {
                    float3 p = sharedCircles[i].position;
                    float rad = sharedCircles[i].radius;
                    
                    // distance check to see if pixel is within circle's radius
                    float diffX = p.x - pixelCenterNorm.x;
                    float diffY = p.y - pixelCenterNorm.y;
                    float pixelDist = diffX * diffX + diffY * diffY;
                    float maxDist = rad * rad;
                    
                    if (pixelDist <= maxDist) {
                        // Circle contributes to pixel
                        float3 rgb = sharedColors[i];
                        float alpha = .5f;
                        
                        if (cuConstRendererParams.sceneName == SNOWFLAKES || 
                            cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
                            const float kCircleMaxAlpha = .5f;
                            const float falloffScale = 4.f;
                            
                            float normPixelDist = sqrt(pixelDist) / rad;
                            rgb = lookupColor(normPixelDist);
                            
                            float maxAlpha = .6f + .4f * (1.f-p.z);
                            maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
                            alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
                        }
                        
                        // blend color into accumulator
                        float oneMinusAlpha = 1.f - alpha;
                        accumColor.x = alpha * rgb.x + oneMinusAlpha * accumColor.x;
                        accumColor.y = alpha * rgb.y + oneMinusAlpha * accumColor.y;
                        accumColor.z = alpha * rgb.z + oneMinusAlpha * accumColor.z;
                        accumColor.w = alpha + accumColor.w;
                    }
                }
                
                // write accumulated color back to global memory
                *imgPtr = accumColor;
            }
        }
        __syncthreads();
    }
}
// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // COMMENT: this definitely does not need to be done sequentially (ther are no inter dependecies)
    // for all pixels in the circle's bounding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr);
            imgPtr++;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }

    if (cudaDeviceIntersectionMatrix) {
        cudaFree(cudaDeviceIntersectionMatrix);
        cudaFree(cudaDeviceTileCounts);
        cudaFree(cudaDeviceTileOffsets);
        cudaFree(cudaDeviceTileCircleLists);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;

    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    // TODO: add allocation of data structures
    // 1. intersection matrix 
    // 2. tile counts
    // 3. tile offsets
    // 4. circle lists (need to determine max size)

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    numTilesX = (image->width + TILE_SIZE - 1) / TILE_SIZE;
    numTilesY = (image->height + TILE_SIZE - 1) / TILE_SIZE;
    totalTiles = numTilesX * numTilesY;
    
    // allocate the intersection matrix and count arrays (256 tiles * 1M circles ~ 256MB)
    cudaCheckError(
        cudaMalloc(&cudaDeviceIntersectionMatrix, 
                   sizeof(char) * totalTiles * numCircles));
    
    // allocate the transposed intersection matrix and count arrays (256 tiles * 1M circles ~ 256MB)
    cudaCheckError(
        cudaMalloc(&cudaDeviceIntersectionMatrixTransposed, 
                   sizeof(char) * numCircles * totalTiles));
    
    cudaCheckError(
        cudaMalloc(&cudaDeviceTileCounts, 
                   sizeof(int) * totalTiles));
    
    cudaCheckError(
        cudaMalloc(&cudaDeviceTileOffsets, 
                   sizeof(int) * (totalTiles + 1)));

    // init arrays
    cudaCheckError(
        cudaMemset(cudaDeviceIntersectionMatrix, 0, 
                  sizeof(char) * totalTiles * numCircles));
    cudaCheckError(
        cudaMemset(cudaDeviceTileCounts, 0, 
                  sizeof(int) * totalTiles));
    cudaCheckError(
        cudaMemset(cudaDeviceTileOffsets, 0, 
                  sizeof(int) * (totalTiles + 1)));

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

void CudaRenderer::render() {
    // clear image
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();

    // phase 1a: compute intersections
    dim3 blockDim1a(256, 1);
    dim3 gridDim1a((numCircles + blockDim1a.x - 1) / blockDim1a.x);
    kernelComputeIntersections<<<gridDim1a, blockDim1a>>>(
        numCircles,
        numTilesX,
        numTilesY,
        cudaDeviceIntersectionMatrix
    );
    cudaDeviceSynchronize();

    // phase 1b: compute tile counts using prefix sum
    dim3 blockDim1b(256, 1);
    dim3 gridDim1b((totalTiles + 7) / 8, 1);
    kernelComputeTileCounts<<<gridDim1b, blockDim1b>>>(
        numCircles,
        numTilesX,
        numTilesY,
        cudaDeviceIntersectionMatrix,
        cudaDeviceTileCounts
    );
    cudaDeviceSynchronize();
    
    // compute offsets using our kernel
    kernelComputeOffsets<<<1, 1>>>(
        cudaDeviceTileCounts,
        cudaDeviceTileOffsets,
        totalTiles
    );
    cudaDeviceSynchronize();
    
    // get total number of intersections
    int totalIntersections;
    cudaCheckError(
        cudaMemcpy(&totalIntersections,
                   &cudaDeviceTileOffsets[totalTiles],
                   sizeof(int),
                   cudaMemcpyDeviceToHost));
    
    // allocate exactly the space we need for tileCircleLists (not worst case)
    if (cudaDeviceTileCircleLists != nullptr) {
        cudaFree(cudaDeviceTileCircleLists);
    }
    cudaCheckError(
        cudaMalloc(&cudaDeviceTileCircleLists,
                   sizeof(int) * totalIntersections));

    // phase 2: build circle lists
    dim3 blockDim2(256, 1);  // must match scan size
    dim3 gridDim2((totalTiles + 7) / 8, 1);

    kernelBuildTileLists<<<gridDim2, blockDim2>>>(
        numCircles,
        cudaDeviceIntersectionMatrix,
        cudaDeviceTileCounts,
        cudaDeviceTileOffsets,
        cudaDeviceTileCircleLists,
        numTilesX,
        numTilesY
    );
    cudaDeviceSynchronize();

    // phase 3: render tiles
    dim3 blockDim3(32, 32);
    dim3 gridDim3(numTilesX, numTilesY);
    
    kernelRenderTiles<<<gridDim3, blockDim3>>>(
        numCircles,
        image->width,
        image->height,
        cudaDeviceTileOffsets,
        cudaDeviceTileCircleLists,
        numTilesX,
        numTilesY
    );
    cudaDeviceSynchronize();
}