
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