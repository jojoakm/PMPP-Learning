#include <cuda_runtime.h>
#include <iostream>

using namespace std;
constexpr int TILE_SIZE = 16;

// A: row-major, shape = (n1, m)
// B: column-major, shape = (m, n2)
// C: row-major, shape = (n1, n2)
// 目标：每个线程计算 C 的一个元素 C[y][x]
__global__ void gemm_corner_turning_kernel(const int* a, const int* b_col, int* c, int n1, int n2, int m) {
    // 当前 block 要计算的 A/B 子块（tile）放到 shared memory
    __shared__ int a_m[TILE_SIZE][TILE_SIZE];
    __shared__ int b_m[TILE_SIZE][TILE_SIZE];

    // 线程对应的全局输出坐标 (y, x)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 沿着 K 维（这里是 m）分块迭代
    int times = (m + TILE_SIZE - 1) / TILE_SIZE;
    int res = 0;

    for (int i = 0; i < times; i++) {
        // ---------- 加载 A tile ----------
        // A 的列索引 = 当前 tile 的起点 + tx
        int a_col = i * TILE_SIZE + threadIdx.x;
        // A 是 row-major：A[y][a_col] -> a[y * m + a_col]
        if (y < n1 && a_col < m) a_m[threadIdx.y][threadIdx.x] = a[y * m + a_col];
        else a_m[threadIdx.y][threadIdx.x] = 0;

        // ---------- 加载 B tile（转角法核心）----------
        // B 的行索引沿 K 维推进，列索引就是输出列 x
        int b_row = i * TILE_SIZE + threadIdx.x;
        int b_col_idx = x;
        // B 是 column-major：B[b_row][b_col_idx] -> b_col[b_col_idx * m + b_row]
        // 注意这里写入 shared memory 用了 b_m[tx][ty]（相当于转置写入）
        // 后续计算阶段可以按 b_m[j][tx] 顺序访问，减少 bank conflict，且便于连续使用
        if (b_row < m && b_col_idx < n2) b_m[threadIdx.x][threadIdx.y] = b_col[b_col_idx * m + b_row];
        else b_m[threadIdx.x][threadIdx.y] = 0;

        // 确保整个 tile 加载完成
        __syncthreads();

        // 用当前 tile 做一次长度为 TILE_SIZE 的点积累加
        for (int j = 0; j < TILE_SIZE; j++) {
            res += a_m[threadIdx.y][j] * b_m[j][threadIdx.x];
        }
        // 防止下一轮 tile 覆盖还在使用的数据
        __syncthreads();
    }

    // 写回结果 C[y][x]
    if (x < n2 && y < n1) c[y * n2 + x] = res;
}

void gemm_corner_turning(const int* a, const int* b_col, int* c, int n1, int n2, int m) {
    int *a_d, *b_d, *c_d;
    cudaMalloc((void**)&a_d, n1 * m * sizeof(int));
    cudaMalloc((void**)&b_d, m * n2 * sizeof(int));
    cudaMalloc((void**)&c_d, n1 * n2 * sizeof(int));

    // Host -> Device
    cudaMemcpy(a_d, a, n1 * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_col, m * n2 * sizeof(int), cudaMemcpyHostToDevice);

    // 2D 网格：每个 block 负责 C 的一个 TILE_SIZE x TILE_SIZE 子块
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n2 + TILE_SIZE - 1) / TILE_SIZE, (n1 + TILE_SIZE - 1) / TILE_SIZE);
    gemm_corner_turning_kernel<<<grid, block>>>(a_d, b_d, c_d, n1, n2, m);

    // Device -> Host
    cudaMemcpy(c, c_d, n1 * n2 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

int main() {
    constexpr int n1 = 2;
    constexpr int m = 3;
    constexpr int n2 = 2;

    int A[n1 * m] = {
        1, 2, 3,
        4, 5, 6
    };

    // B（列主序存储）对应矩阵（逻辑上）：
    // [ 7  8 ]
    // [ 9 10 ]
    // [11 12 ]
    // 列主序展开 = [7, 9, 11, 8, 10, 12]
    int B_col[m * n2] = {7, 9, 11, 8, 10, 12};

    int C[n1 * n2] = {0};
    gemm_corner_turning(A, B_col, C, n1, n2, m);

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) cout << C[i * n2 + j] << " ";
        cout << endl;
    }
    return 0;
}
