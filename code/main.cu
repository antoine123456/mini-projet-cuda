#include <iostream>
#include <string.h>
#include "cuda_runtime.h"
#include <sstream>
#include <fstream>
#include "FreeImage.h"
#define WIDTH 10
#define HEIGHT 10
// #define WIDTH 3840
// #define HEIGHT 2160
#define BPP 24 // Since we're outputting three 8 bit RGB values

using namespace std;

int main(int argc, char **argv)
{
    FreeImage_Initialise();
    const char *PathName = "img.jpg";
    const char *PathDest = "new_img.png";
    // load and decode a regular file
    FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(PathName);

    FIBITMAP *bitmap = FreeImage_Load(FIF_JPEG, PathName, 0);

    if (!bitmap)
        exit(1); // WTF?! We can't even allocate images ? Die !

    unsigned width = FreeImage_GetWidth(bitmap);
    unsigned height = FreeImage_GetHeight(bitmap);
    unsigned pitch = FreeImage_GetPitch(bitmap);
    const int imageSize = width * height * 3;
    const int color = 0; // use green channel

    fprintf(stderr, "Processing Image of size %d x %d\n", width, height);

    unsigned int *img = (unsigned int *)malloc(sizeof(unsigned int) * 3 * width * height);

    // stockage de l'image
    ofstream saveFormat;
    saveFormat.open("bimage.bin", std::ios::out | std::ios::binary);
    BYTE *bits = (BYTE *)FreeImage_GetBits(bitmap);
    for (int y = 0; y < height; y++)
    {
        BYTE *pixel = (BYTE *)bits;
        for (int x = 0; x < width; x++)
        {
            int idx = ((y * width) + x) * 3;
            img[idx + 0] = pixel[FI_RGBA_RED];
            img[idx + 1] = pixel[FI_RGBA_GREEN];
            img[idx + 2] = pixel[FI_RGBA_BLUE];
            pixel += 3;
            saveFormat << img[idx + 0] << "," << img[idx + 1] << "," << img[idx + 2] << ",";
        }
        // next line
        bits += pitch;
    }

    // allocate memory for input and output images on GPU
    unsigned int *d_img, *d_outputImage;
    cudaMalloc((void **)&d_img, imageSize * sizeof(unsigned int));
    cudaMalloc((void **)&d_outputImage, imageSize * sizeof(unsigned int));

    // copy input image to GPU
    cudaMemcpy(d_img, img, imageSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // set up kernel launch parameters
    dim3 block(32, 32, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);

    // call kernel
    saturateColor<<<grid, block>>>(d_img, d_outputImage, width, height, color, 100.0f);
    invertHorizontally<<<grid, block>>>(d_img, d_outputImage, width, height);
    sobelFilter<<<grid, block>>>(d_img, d_outputImage, width, height);
    greyScaleTransform<<<grid, block>>>(d_img, d_outputImage, width, height);
    blurImage<<<grid, block>>>(d_img, d_outputImage, width, height);
    createFourSubImages<<<grid, block>>>(d_img, d_outputImage, width, height);

    // copy output image from GPU to host
    cudaMemcpy(img, d_outputImage, imageSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    bits = (BYTE *)FreeImage_GetBits(bitmap);
    for (int y = 0; y < height; y++)
    {
        BYTE *pixel = (BYTE *)bits;
        for (int x = 0; x < width; x++)
        {
            RGBQUAD newcolor;

            int idx = ((y * width) + x) * 3;
            newcolor.rgbRed = img[idx + 0];
            newcolor.rgbGreen = img[idx + 1];
            newcolor.rgbBlue = img[idx + 2];

            if (!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
            {
                fprintf(stderr, "(%d, %d) Fail...\n", x, y);
            }

            pixel += 3;
        }
        // next line
        bits += pitch;
    }

    if (FreeImage_Save(FIF_PNG, bitmap, PathDest, 0))
        cout << "\nImage successfully saved ! " << endl;
    FreeImage_DeInitialise(); // Cleanup !

    free(img);
}
