/// @brief Saturates the color of an image
/// @param inputImage Pointer to the input image
/// @param outputImage Pointer to the output image
/// @param width Width of the image
/// @param height Height of the image
/// @param color Color to saturate (0 = red, 1 = green, 2 = blue)
/// @param saturationFactor Factor to saturate the color by
/// @return void
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