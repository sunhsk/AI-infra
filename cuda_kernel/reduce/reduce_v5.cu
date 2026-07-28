#include<stdio.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Warp 内规约辅助函数
__device__ float warpReduceSum(float val) {
    // 每次将右半边的值加到左半边
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // lane 0 持有最终结果
}

// V6: Warp Shuffle + 两级规约
__global__ void reduce_v5(float* input, float* output, int n) {
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane = tid % 32;      // 线程在 Warp 内的编号（0~31）
    int wid  = tid / 32;      // 该线程属于哪个 Warp

    // 每线程处理 2 个元素
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];

    // 第一级：Warp 内规约
    val = warpReduceSum(val);

    // 将每个 Warp 的结果（仅 lane 0 有效）存入 Shared Memory
    __shared__ float warp_results[32];  // 最多 32 个 Warp（1024/32）
    if (lane == 0) {
        warp_results[wid] = val;
    }
    __syncthreads();

    // 第二级：Warp 间规约（用 Warp 0 处理）
    int num_warps = blockDim.x / 32;                     //计算当前 Block 内一共有多少个 Warp
    if (wid == 0) {                                      // 只用Block内第0号Warp来合并所有warp结果
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    if (tid == 0) output[blockIdx.x] = val;
}




int main(void){
    int n = 1 << 20;
    size_t size = n * sizeof(float);

    float* h_input = (float*)malloc(size);

    if (h_input == nullptr)
    {
        fprintf(stderr, "Failed to allocate host input\n");
        return -1;
    }

    float h_output = 0.0f;

    // Initialize input data
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f; // For simplicity, initialize all elements to 1.0f
    }

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA(cudaMalloc((void**)&d_output, size));


    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int numElements = n ;
    float* d_in = d_input;
    float* d_out = d_output;

    // 每一轮把当前输入规约成 blocksPerGrid 个部分和。
    // 反复执行直到 device 上只剩 1 个 float。
    while (numElements > 1) {
        int blocksPerGrid = (numElements + 2 * threadsPerBlock - 1) / (2 * threadsPerBlock);

        // 第三个参数是共享内存大小：每个线程一个 float。
        
        reduce_v5<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_in, d_out, numElements);  //threadsPerBlock * sizeof(float) 是 CUDA kernel 启动时指定的动态共享内存大小，单位是字节。每个 block 分配的动态共享内存大小
        CHECK_CUDA(cudaGetLastError());

        numElements = blocksPerGrid;

        float* tmp = d_in;
        d_in = d_out;
        d_out = tmp;                                                       // 复用另一块显存作为输出
    }

    // 最终结果已经在 device 上规约成 1 个 float，只需要拷回 sizeof(float)。
    CHECK_CUDA(cudaMemcpy(&h_output, d_in, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Result: %f\n", h_output); // Should print 1048576.0 (1M)

    free(h_input);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    

    return 0;
}