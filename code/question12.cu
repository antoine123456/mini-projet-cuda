/**
 * @brief Creates four sub-images from an input image
 * 
 * @param inputImage The input image
 * @param outputImage The output image
 * @param width The width of the input image
 * @param height The height of the input image
 */
__global__ void createFourSubImages(unsigned int *inputImage, unsigned int *outputImage, int width, int height) {
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