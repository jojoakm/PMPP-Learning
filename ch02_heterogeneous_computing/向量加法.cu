#include<iostream>

using namespace std;
 //这里有device host，global 三种关键字，就是global就是核函数，host是默认的，devcice是gpu，也可以host，device都有
 //如果是device host都有那就是编译两份一份属于cpu，一份属于gpu
__global__ void vectorkernaladd(int *a,int *b,int *c,int n)//定义核函数，就是在gpu上并行进行向量加法。
{
    //在核函数内部是只能使用gpu的部件的。
    //一般来说申请gpu的网格都是在gpu外部进行的
    //所以说这里是已经申请完成了
    //还有就是当前这个核函数会平等的给到每一个gpu内部的线程然后每个线程都有内置变量负责展示自己的id
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n) c[i] = a[i] +b[i]; //这里就是说每一个线程如果他的编号是在范围内的就把对应变量的向量内部的元素进行计算完毕。
}

void vectoradd(int *a,int *b,int *c,int n)
{
    //还要注意就是我们申请空间的时候申请到的是字节的数据，所以说要把空间转换成字节空间
    int size = n * sizeof(int);

    int *a_d;
    int *b_d;
    int *c_d; //这三个指针往往是不预分配的，是用来存放分配完存储的gpu上的指针的。
    
    //首先是要在gpu上开辟空间
    //其写法比较类似malloc
     cudaMalloc((void**)&a_d, size);
     cudaMalloc((void**)&b_d, size);
     cudaMalloc((void**)&c_d, size);

     //申请完空间之后我们应当计算网格和线程块的大小

     //假设我们这里用的是一个线程块内部256个线程的配置，计算一共有多少个线程块
     int block_dim = ceil(n/256.0);//这里需要是上取整，注意并不是一个线程对应一个字节

     //接下来我们应当把cpu主机的数据复制到gpu上面 注意这个指针是先说目标后说源头
      cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
      cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);

   //这里有device host，global 三种关键字，就是global就是核函数，host是默认的，devcice是gpu，也可以host，device都有
      vectorkernaladd <<<block_dim,256>>>(a_d,b_d,c_d,n);//不同的关键字就是编译之后的处理不一样，host是cpu编译，devicegpu编译，在后面组合成exe
      //如果是device host都有那就是编译两份一份属于cpu，一份属于gpu

      cudaMemcpy(c, c_d, size, cudaMemcpyDeviceToHost);//这里是把gpu上的数据复制到cpu主机上

      cudaFree(a_d);
      cudaFree(b_d);
      cudaFree(c_d);

}

int main()
{
    int n = 10;
    int *a = new int[n];
    int *b = new int[n];
    int *c = new int[n];
    for(int i=0;i<n;i++)a[i] = i;
    for(int i=0;i<n;i++)b[i] = i;
    vectoradd(a,b,c,n);
    for(int i=0;i<n;i++)cout<<c[i]<<endl;
    cout<<endl;
    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}