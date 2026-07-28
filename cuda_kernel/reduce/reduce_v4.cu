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

// 辅助函数：展开最后 32 个线程的规约
__device__ void warpReduce(volatile float* smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid +  8];
    smem[tid] += smem[tid +  4];
    smem[tid] += smem[tid +  2];
    smem[tid] += smem[tid +  1];
}

// V4: 展开最后一个 Warp（在 V3 基础上）
__global__ void reduce_v4(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 每线程处理 2 个元素（继承自 V3）
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    // 规约循环仅执行到 step > 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // 最后一个 Warp 内的规约，无需 __syncthreads()
    if (tid < 32) {
        warpReduce(smem, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
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

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 每一轮把当前输入规约成 blocksPerGrid 个部分和。
    // 反复执行直到 device 上只剩 1 个 float。
    while (numElements > 1) {
        int blocksPerGrid = (numElements + 2 * threadsPerBlock - 1) / (2 * threadsPerBlock);

        // 第三个参数是共享内存大小：每个线程一个 float。
        
        reduce_v4<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_in, d_out, numElements);  //threadsPerBlock * sizeof(float) 是 CUDA kernel 启动时指定的动态共享内存大小，单位是字节。每个 block 分配的动态共享内存大小
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
    

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Reduction time: %.6f ms\n", milliseconds);
    return 0;
}