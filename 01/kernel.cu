#include <cstdio>

extern "C" __global__ void voronoi(int* cell, int width, int height, int n_centers, int* centers) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int threadIndex = x + y * width;

    // For 32-bit ints, this is 2**31.
    // This will not overflow if our map is smaller than 32768 x 32768:
    //     2 x^2 <= 2^31
    // <=> x <= sqrt(2^30) == 2^15 == 32768
    int closest_dist_sq = INT_MAX;

    // Iterate through all the centers and choose the closest one.
    int closest_center = -1;
    for(int ic = 0; ic < n_centers; ic++) {
        int cx = centers[ic * 2];
        int cy = centers[ic * 2 + 1];

        int dist_sq = (cx - x) * (cx - x) + (cy - y) * (cy - y);

        if(dist_sq < closest_dist_sq) {
            closest_center = ic;
            closest_dist_sq = dist_sq;
        }
    }

    cell[threadIndex] = closest_center;
}