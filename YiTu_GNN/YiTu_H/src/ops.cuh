__device__ inline float leaky_relu(float x, float negative_slope) {
    return fmaxf(0.0, x) + negative_slope * fminf(0.0, x);
}

__device__ inline float grad_leaky_relu(float x, float negative_slope) {
    if (x < 0.0) {
        return negative_slope;
    }
    return 1.0;
}
