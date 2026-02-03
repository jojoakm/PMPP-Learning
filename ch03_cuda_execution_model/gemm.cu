#include<iostream>

using namespace std;

__global__ 
void gemm_kernal(float* a,float* b,float* c,int m1,int n,int m2)
{
    //a，b，c是三个矩阵的第一个数值的gpu地址
    //a m1xn  b nxm2  c m1xm2
    //首先要计算出当前元素的坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //这是我们的计算的那个点的x，y
    //首先要排除特殊情况
    
    //注意这里x的极限是m2,y的极限是m1
    if(x >= m2 || y >= m1)return;

    //接下来就是a矩阵第x行数据和b矩阵第y列搞一个点积就可以大功告成啦芜湖

    float res = 0;
    int a_index;
    int b_index;
    for(int i=0;i<n;i++)
    {
        //接下来计算出a矩阵第y行第i个数据和b矩阵第x列第i个数据的
        //keke直接计算指针简单是简单用起来不舒服这里返回索引
        a_index = y*n+i;
        b_index = i*m2+x;
        res += a[a_index] * b[b_index];
    }
    c[y*m2+x] = res;
    //至此就把计算出来的结果放到c矩阵的x行y列
}


void gemm(float* a,float* b,float* c,int m1,int n,int m2)
{
    //a矩阵的形状 m1xn   b矩阵的形状 nxm2  c矩阵的形状  m1xm2
    //这里就不为了简单搞方阵了，我们直接弄最真实的

    //第一步要分配gpu的内存资源
    float* a_d;
    float* b_d;
    float* c_d;
    cudaMalloc((void**)&a_d,m1*n*sizeof(float));
    cudaMalloc((void**)&b_d,n*m2*sizeof(float));
    cudaMalloc((void**)&c_d,m1*m2*sizeof(float));

    //第二步要拷贝数据到gpu
    //之前犯过的错，c不需要拷贝
    cudaMemcpy(a_d,a,m1*n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,n*m2*sizeof(float),cudaMemcpyHostToDevice);

    //第三步要定义我们核函数要开多少线程
    //每个线程负责计算c矩阵的一个元素
    //这里假如说我们用16x16的线程块
    //横向就需要ceil（m1/16）个块
    //纵向就需要ceil (m2/16) 个块
    //然后用dim3分配的时候是反过来的

    //我们的最后的矩阵是m1 x m2   这个时候 x对应是m2  y对应是m1
    dim3 dealwithblock(ceil(m2/16.0),ceil(m1/16.0));//这里是块的维度二维的话就是x,y
    dim3 dealwiththread(16,16);//这里是块内部的线程xy

    //接下来调用核函数
    gemm_kernal<<<dealwithblock,dealwiththread>>>(a_d,b_d,c_d,m1,n,m2);

    //接下来是拷贝结果回cpu
    cudaMemcpy(c,c_d,m1*m2*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
} 

int main()
{
    int m1,m2,n;
    cin>>m1>>m2>>n;
    float *a_h = new float[m1*n];
    float *b_h = new float[n*m2];
    float *c_h = new float[m1*m2];
    for(int i=0;i<m1*n;i++)cin>>a_h[i];
    for(int i=0;i<n*m2;i++)cin>>b_h[i];
    gemm(a_h,b_h,c_h,m1,n,m2);
    for(int i=0;i<m1;i++)
    {
        for(int j=0;j<m2;j++)cout<<c_h[i*m2+j]<<" ";
        cout<<endl;
    }

    return 0;
}