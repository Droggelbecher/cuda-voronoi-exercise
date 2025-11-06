#include <cstdio>

extern "C" __global__ void voronoi(int* cell, int width, int height, int n_centers, int* centers) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int i = x + y * width;

    cell[i] = 0;

    int closest_center = -1;
    int closest_dist_sq = INT_MAX;

    for(int ic = 0; ic < n_centers; ic++) {
        int cx = centers[ic * 2];
        int cy = centers[ic * 2 + 1];

        int dist_sq = (cx - x) * (cx - x) + (cy - y) * (cy - y);

        if(dist_sq < closest_dist_sq) {
            closest_center = ic;
            closest_dist_sq = dist_sq;
        }
    }

    cell[i] = closest_center;
}