
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