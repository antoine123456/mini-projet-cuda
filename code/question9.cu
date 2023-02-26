   /**
    * @brief Performs a greyscale transformation on an image
    * 
    * @param inputImage The input image
    * @param outputImage The output image
    * @param width The width of the image
    * @param height The height of the image
    */
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