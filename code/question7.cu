/**
* Inverts an image horizontally.
* 
* @param inputImage The input image.
* @param outputImage The output image.
* @param width The width of the image.
* @param height The height of the image.
*/
__global__ void invertHorizontally(unsigned int *inputImage, unsigned int *outputImage, int width, int height) {
    // Get the thread and block indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if the indices are within the image bounds
    if (i < width && j < height) {
        // Calculate the index of the current pixel
        int index = j * width * 3 + i * 3;
        // Calculate the index of the corresponding pixel in the output image
        int newIndex = j * width * 3 + (width - i - 1) * 3;
        // Copy the pixel values from the input image to the output image
        outputImage[newIndex] = inputImage[index];
        outputImage[newIndex + 1] = inputImage[index + 1];
        outputImage[newIndex + 2] = inputImage[index + 2];
    }
}