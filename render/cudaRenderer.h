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

    // ADDED

    static const int TILE_SIZE = 64;
    int numTilesX;
    int numTilesY;
    int totalTiles;

    char *cudaDeviceIntersectionMatrix;           // binary matrix [numTiles x numCircles]
    char *cudaDeviceIntersectionMatrixTransposed; // [numCircles x numTiles]
    int *cudaDeviceTileCounts;                    // count of circles per tile
    int *cudaDeviceTileOffsets;                   // prefix sum of tileCounts (for indexing)
    int *cudaDeviceTileCircleLists;               // final ordered circle lists per tile

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
