#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"

class CudaRenderer : public CircleRenderer
{

private:
    Image *image;
    SceneName sceneName;

    int numCircles;
    float *position;
    float *velocity;
    float *color;
    float *radius;

    float *cudaDevicePosition;
    float *cudaDeviceVelocity;
    float *cudaDeviceColor;
    float *cudaDeviceRadius;
    float *cudaDeviceImageData;

    // added
    char *cudaDeviceIntersectionMatrix; // circle-tile intersection matrix (flattened numTilesxnumCircles array)
    int *cudaDeviceTileCounts;          // number of circles intersecting each tile
    int *cudaDeviceTileOffsets;         // starting index in TileCircleLists for each tile
    int *cudaDeviceTileCircleLists;     // array storing circle indices for each tile

    int numTilesX;
    int numTilesY;
    int totalTiles;

public:
    CudaRenderer();
    virtual ~CudaRenderer();

    const Image *getImage();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();

    void shadePixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float *pixelData);
};

#endif
