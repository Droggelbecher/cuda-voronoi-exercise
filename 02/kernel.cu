#include <cstdio>

extern "C" __global__ void initialize(int *cell, int width, int height, int n_centers, int *centers)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int threadIndex = x + y * width;

    cell[threadIndex] = -1;
}


extern "C" __global__ void set_voronoi_sites(int *cell, int width, int height, int n_centers, int *centers)
{
    // 1. Initialization:
    // If we are at a voronoi site, initialize to that index, otherwise -1

    int ic = blockIdx.x * blockDim.x + threadIdx.x;

    if(ic >= n_centers)
        return;

    int cx = centers[ic * 2];
    int cy = centers[ic * 2 + 1];

    cell[cx + cy * width] = ic;
}

extern "C" __global__ void voronoi(int *cell, int width, int height, int n_centers, int *centers, int d)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int threadIndex = x + y * width;

    // 2. Jump Flooding:
    // Repeatedly check 8 neighbours at increasingly closer distances

    // For 32-bit ints, this is 2**31.
    // This will not overflow if our map is smaller than 32768 x 32768:
    //     2 x^2 <= 2^31
    // <=> x <= sqrt(2^30) == 2^15 == 32768
    int closest_dist_sq = INT_MAX;

    // Iterate through all the centers and choose the closest one.
    int closest_site = cell[threadIndex];
    if(closest_site != -1) {
        int cx = centers[closest_site * 2];
        int cy = centers[closest_site * 2 + 1];
        closest_dist_sq = (cx - x) * (cx - x) + (cy - y) * (cy - y);
    }

    // (dx, dy) is relative position of the neighbor to check
    for (int dy = -d; dy <= d; dy += d)
    {
        if (y + dy < 0 || y + dy >= height)
            continue;

        for (int dx = -d; dx <= d; dx += d)
        {
            if (x + dx < 0 || x + dx >= width)
                continue;

            if (dx == 0 && dy == 0)
                continue;

            int neighborIndex = (x + dx) + (y + dy) * width;
            int ic = cell[neighborIndex];

            if (ic != -1)
            {
                // Our neighbor at distance (dx, dy) has

                int cx = centers[ic * 2];
                int cy = centers[ic * 2 + 1];
                int dist_sq = (cx - x) * (cx - x) + (cy - y) * (cy - y);
                if (dist_sq < closest_dist_sq)
                {
                    closest_dist_sq = dist_sq;
                    closest_site = ic;
                }
            }
        }
    }

    if(closest_site != -1)
        cell[threadIndex] = closest_site;
}