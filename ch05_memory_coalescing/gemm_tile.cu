#include <iostream>
#include <cuda_runtime.h> // 必须包含 CUDA 运行时头文件
#include <cmath>        // 必须包含此头文件，否则 ceil 编不过去

using namespace std;
constexpr int TILE_SIZE = 16;

__global__
void gemm_kernal(int* a, int* b, int* c, int n1, int n2, int m)
{
    __shared__ int a_m[TILE_SIZE][TILE_SIZE];
    __shared__ int b_m[TILE_SIZE][TILE_SIZE];

    // 逻辑没动，修正了大小写，这是 CUDA 的固定变量名
    int x = blockDim.x * blockIdx.x + threadIdx.x; 
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int times = (m + TILE_SIZE - 1) / TILE_SIZE;
    int res = 0;
    for(int i=0;i<times;i++)
    {
        // A 是 row-major（n1 x m）：
        // a[y*m + (i*TILE_SIZE + tx)] 让同一个 warp 的 tx 连续递增，
        // 读取地址连续，global memory 更容易 coalesced。
        if(y>=n1||threadIdx.x+i*TILE_SIZE>=m)a_m[threadIdx.y][threadIdx.x] = 0;
        else a_m[threadIdx.y][threadIdx.x] = a[y*m+threadIdx.x+i*TILE_SIZE];

        // B 在这个文件里按 row-major（m x n2）：
        // b[(i*TILE_SIZE + ty)*n2 + x] 同理让 x 连续时地址连续，便于 coalesced。
        //
        // 你问“第二个矩阵如果按列存储是不是很难受”：是的，这里会别扭。
        // 若 B 改成 column-major，直接写成 b[x*m + (i*TILE_SIZE + ty)]，
        // 通常会让当前线程映射下的访存模式变差（不够连续/需要改线程映射），
        // 所以常用做法是第六章那种 corner turning：加载到 shared 时做一次转置映射。
        if (x>=n2||threadIdx.y+i*TILE_SIZE>=m)b_m[threadIdx.y][threadIdx.x] = 0;
        else b_m[threadIdx.x][threadIdx.y] = b[threadIdx.x+i*tile_size][blockdim.x*blockidx.x+threadIdx.y] =  b[(blockdim.x*blockidx.x+threadIdx.y)*m+threadIdx.x+i*tile_size];
        __syncthreads();

        // 这里用 shared memory 做 tile 内积，减少重复访问全局内存。
        for(int j=0;j<TILE_SIZE;j++)
        {
            res+=a_m[threadIdx.y][j]*b_m[j][threadIdx.x];
        }
        __syncthreads();
    }
    if(x<n2&&y<n1)c[y*n2+x] = res;
        
}

void gemm(int* a, int* b, int* c, int n1, int n2, int m)
{
    int *a_d, *b_d, *c_d; // 按照你的要求，全部改为 _d

    // 分配显存
    cudaMalloc((void**)&a_d, n1 * m * sizeof(int));
    cudaMalloc((void**)&b_d, m * n2 * sizeof(int)); // 修正逻辑：B 是 m*n2
    cudaMalloc((void**)&c_d, n1 * n2 * sizeof(int));

    // 拷贝数据：必须加上方向标识，否则运行会出错
    cudaMemcpy(a_d, a, n1 * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, m * n2 * sizeof(int), cudaMemcpyHostToDevice);

    // 计算 Grid 大小
    dim3 d2(TILE_SIZE, TILE_SIZE);
    // 这里 ceil 的用法对，但要确保除以 16.0 浮点数以保证精度
    dim3 d1((n2 + TILE_SIZE - 1) / TILE_SIZE, (n1 + TILE_SIZE - 1) / TILE_SIZE);

    gemm_kernal<<<d1, d2>>>(a_d, b_d, c_d, n1, n2, m);

    // 拷回数据
    cudaMemcpy(c, c_d, n1 * n2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // --- 这里是打印逻辑的问题点 ---
    // 原代码用了 i < m，如果不是方阵会打错。建议改成输出矩阵的大小 n1 * n2
    for(int i = 0; i < n1; i++) 
    {
        for(int j = 0; j < n2; j++) cout << c[i * n2 + j] << " ";
        cout << endl;
    }

    // 战神建议：加一下 cudaFree，防止 Nsight 分析时显存溢出
    cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);
}

int main()
{
    // A: n1 x m, B: m x n2, C: n1 x n2
    constexpr int n1 = 2;
    constexpr int m = 3;
    constexpr int n2 = 2;

    int A[n1 * m] = {
        1, 2, 3,
        4, 5, 6
    };

    int B[m * n2] = {
         7,  8,
         9, 10,
        11, 12
    };

    int C[n1 * n2] = {0};

    gemm(A, B, C, n1, n2, m);
    return 0;
}
