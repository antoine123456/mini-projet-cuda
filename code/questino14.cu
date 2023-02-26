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

__global__ void saturateColor(unsigned int *inputImage, unsigned int *outputImage, int width, int height, int color, float saturationFactor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height)
    {
        int index = j * width * 3 + i * 3;
        float value = inputImage[index + color] / 255.0f;
        value = value * saturationFactor;
        value = min(value, 1.0f);
        value = max(value, 0.0f);
        outputImage[index + color] = (unsigned int)(value * 255.0f);
        for (int k = 0; k < 3; k++)
        {
            if (k != color)
            {
                outputImage[index + k] = inputImage[index + k];
            }
        }
    }
}
/**
* Inverts an image horizontally.
* 
* @param inputImage The input image.
* @param outputImage The output image.
* @param width The width of the image.
* @param height The height of the image.
*/
__global__ void invertHorizontally(unsigned int *inputImage, unsigned int *outputImage, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < height) {
        int index = j * width * 3 + i * 3;
        int newIndex = j * width * 3 + (width - i - 1) * 3;
        outputImage[newIndex] = inputImage[index];
        outputImage[newIndex + 1] = inputImage[index + 1];
        outputImage[newIndex + 2] = inputImage[index + 2];
    }
}
__global__ void greyScaleTransform(unsigned int *inputImage, unsigned int *outputImage, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < width && j < height) {
        int index = j * width * 3 + i * 3;
        float red = inputImage[index + 0];
        float green = inputImage[index + 1];
        float blue = inputImage[index + 2];
        float grey = 0.299f * red + 0.587f * green + 0.114f * blue;
        outputImage[index + 0] = outputImage[index + 1] = outputImage[index + 2] = (unsigned char) grey;
    }
}

__global__ void sobelFilter(unsigned int *inputImage, unsigned int *outputImage, int width, int height) {
    // Calculate the indices of the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Make sure we're not trying to read or write outside the image bounds
    if (x >= width || y >= height) {
        return;
    }

    // Compute the gradients in the x and y directions
    float gx = 0.0f;
    float gy = 0.0f;

    // Apply the Sobel filter to each color channel separately
    for (int channel = 0; channel < 3; ++channel) {
        // Compute the indices of the neighboring pixels
        int indexTopLeft = ((y - 1) * width + (x - 1)) * 3 + channel;
        int indexTop = ((y - 1) * width + x) * 3 + channel;
        int indexTopRight = ((y - 1) * width + (x + 1)) * 3 + channel;
        int indexLeft = (y * width + (x - 1)) * 3 + channel;
        int indexRight = (y * width + (x + 1)) * 3 + channel;
        int indexBottomLeft = ((y + 1) * width + (x - 1)) * 3 + channel;
        int indexBottom = ((y + 1) * width + x) * 3 + channel;
        int indexBottomRight = ((y + 1) * width + (x + 1)) * 3 + channel;

        // Compute the gradients in the x and y directions for this channel
        gx += inputImage[indexTopLeft] * -1.0f + inputImage[indexTopRight] * 1.0f +
              inputImage[indexLeft] * -2.0f + inputImage[indexRight] * 2.0f +
              inputImage[indexBottomLeft] * -1.0f + inputImage[indexBottomRight] * 1.0f;
        gy += inputImage[indexTopLeft] * -1.0f + inputImage[indexTop] * -2.0f + inputImage[indexTopRight] * -1.0f +
              inputImage[indexBottomLeft] * 1.0f + inputImage[indexBottom] * 2.0f + inputImage[indexBottomRight] * 1.0f;
    }

    // Compute the magnitude of the gradient
    float magnitude = sqrtf(gx * gx + gy * gy);

    // Clamp the magnitude to the range [0, 255]
    magnitude = fmaxf(0.0f, fminf(magnitude, 255.0f));

    // Copy the magnitude to the output pixel for each color channel
    int outputIndex = (y * width + x) * 3;
    outputImage[outputIndex] = (unsigned char)magnitude;
    outputImage[outputIndex + 1] = (unsigned char)magnitude;
    outputImage[outputIndex + 2] = (unsigned char)magnitude;
}

__global__ void createFourSubImages(unsigned int *inputImage, unsigned int *outputImage, int width, int height) {
    // Calculate the output image dimensions (half the input dimensions)
    int outputWidth = width / 2;
    int outputHeight = height / 2;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        int index = j * width * 3 + i * 3;
        float red = inputImage[index + 0];
        float green = inputImage[index + 1];
        float blue = inputImage[index + 2];
        float grey = 0.299f * red + 0.587f * green + 0.114f * blue;
        int outIndexGrey = ((j/2)* width * 3 + i/2 * 3);
        outputImage[outIndexGrey] = outputImage[outIndexGrey + 1] = outputImage[outIndexGrey + 2] = (unsigned char) grey; 
        int outIndexR = ((j/2+height/2)* width * 3 + i/2 * 3);
        outputImage[outIndexR] = inputImage[index + 0];
        outputImage[outIndexR + 1] = inputImage[index + 1];
        outputImage[outIndexR + 2] = 255;
        int outIndexG = ((j/2+height/2)* width * 3 + (i/2+width/2) * 3);
        outputImage[outIndexG] = inputImage[index + 0];
        outputImage[outIndexG + 1] = 255;
        outputImage[outIndexG + 2] = inputImage[index + 2];
        int outIndexB = j/2* width * 3 + (i/2+width/2) * 3;
        outputImage[outIndexB] = 255;
        outputImage[outIndexB + 1] = inputImage[index +1];
        outputImage[outIndexB + 2] = inputImage[index + 2];
    }
}

__global__ void blurImage(unsigned int *inputImage, unsigned int *outputImage, int width, int height) {
    // Calculate the indices of the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Make sure we're not trying to read or write outside the image bounds
    if (x >= width || y >= height) {
        return;
    }

    // Compute the average color for each channel
    float red = 0.0f;
    float green = 0.0f;
    float blue = 0.0f;
    int numNeighbors = 0;
  int blurAmp = 20;
    for (int offsetY = -blurAmp; offsetY <= blurAmp; ++offsetY) {
        for (int offsetX = -blurAmp; offsetX <= blurAmp; ++offsetX) {
            // Compute the indices of the neighboring pixels
            int neighborX = x + offsetX;
            int neighborY = y + offsetY;
            if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
                int neighborIndex = (neighborY * width + neighborX) * 3;
                red += inputImage[neighborIndex];
                green += inputImage[neighborIndex + 1];
                blue += inputImage[neighborIndex + 2];
                ++numNeighbors;
            }
        }
    }

    // Compute the average color
    red /= numNeighbors;
    green /= numNeighbors;
    blue /= numNeighbors;

    // Copy the average color to the output pixel for each color channel
    int outputIndex = (y * width + x) * 3;
    outputImage[outputIndex] = (unsigned char)red;
    outputImage[outputIndex + 1] = (unsigned char)green;
    outputImage[outputIndex + 2] = (unsigned char)blue;
}

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

  // b_ stockage de l'image
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
    cudaMalloc((void**)&d_img, imageSize * sizeof(unsigned int));
    cudaMalloc((void**)&d_outputImage, imageSize * sizeof(unsigned int));

    // copy input image to GPU
    cudaMemcpy(d_img, img, imageSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // set up kernel launch parameters
    dim3 block(32, 32, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);

    cudaStream_t stream[4];
    for(int i=0;i<4;i++){
        cudaStreamCreate(&stream[i]);
    }

    int threadsPerBlock = 16;
    dim3 numBlocks((width + threadsPerBlock - 1) / threadsPerBlock, (height + threadsPerBlock - 1) / threadsPerBlock);

    // Define the start and end positions for each stream
    int start[4], end[4];
    for (int i = 0; i < 4; i++) {
        start[i] = i * height / 4;
        end[i] = (i+1) * height / 4;
    }

    for (int i = 0; i < 4; i++) {
        int size = (end[i] - start[i]) * width * 3 * sizeof(unsigned int);
        unsigned int *d_img, *outputImage_dev;
        cudaMalloc((void **) &d_img, size);
        cudaMalloc((void **) &outputImage_dev, size);
        
        cudaMemcpyAsync(d_img + start[i] * width * 3, img + start[i] * width * 3, size/4, cudaMemcpyHostToDevice, stream[i]);
        
        // Execute kernel on each stream
        createFourSubImages<<<numBlocks, threadsPerBlock, 0, stream[i]>>>(d_img, outputImage_dev, width, height/4);
        
        cudaMemcpyAsync(img + start[i] * width * 3, outputImage_dev + start[i] * width * 3, size/4, cudaMemcpyDeviceToHost, stream[i]);

        cudaFree(d_img);
        cudaFree(outputImage_dev);
    }

    for(int i=0;i<4;i++){
        cudaStreamDestroy(stream[i]);
    }

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
