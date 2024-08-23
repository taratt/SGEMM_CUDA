inline __device__ __host__ size_t ceil_div(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
