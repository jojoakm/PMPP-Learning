#include <iostream>
#include <cuda_runtime.h> // 必须包含 CUDA 运行时头文件
#include <cmath>        // 必须包含此头文件，否则 ceil 编不过去

using namespace std;

__global__
void gemm_kernal(int* a, int* b, int* c, int n1, int n2, int m)
{
    // 逻辑没动，修正了大小写，这是 CUDA 的固定变量名
    int x = blockDim.x * blockIdx.x + threadIdx.x; 
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= n2 || y >= n1) return;

    int total = 0;
    for(int i = 0; i < m; i++)
    {
        // 这里的逻辑很正，y * m 是找 A 的行，i * n2 是找 B 的列
        total += a[y * m + i] * b[i * n2 + x];
    }
    c[y * n2 + x] = total;
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
    dim3 d2(16, 16);
    // 这里 ceil 的用法对，但要确保除以 16.0 浮点数以保证精度
    dim3 d1(ceil(n2 / 16.0), ceil(n1 / 16.0));

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
