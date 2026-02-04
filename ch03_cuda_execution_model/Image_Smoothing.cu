//最后一个作业 做掉平滑图像这最后一个作业！！！！

#include<iostream>

using namespace std;

__global__
void image_smoothing_kernal(int row,int lie,int channel_num,int l,float* pic_d,float* result_d)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x>=lie || y>=row)return;

    //重头戏来了
    int num = 0;
    for(int k=0;k<channel_num;k++)
    {
        result_d[(y*lie+x)*channel_num+k]=0;
    }

    for(int i = -1*l; i <= l; i++)
    {
        for(int j = -1*l; j<= l; j++)
        {
            int y1 = y + i;
            int x1 = x + j;
            if(x1<0||x1>=lie||y1<0||y1>=row)continue;
            num++;
            int index = y1 * lie + x1;
            
            for(int k=0;k<channel_num;k++)
            {
                result_d[(y*lie+x)*channel_num+k]+=pic_d[index*channel_num+k];
            }
        }
    }
    if(num == 0)
    {
        for(int k=0;k<channel_num;k++)
        {
            result_d[(y*lie+x)*channel_num+k]=0;
        } 
    }
    else 
    {
        for(int k=0;k<channel_num;k++)
        {
            result_d[(y*lie+x)*channel_num+k]/=num;
        } 
    }

}

void image_smoothing(int row,int lie,int channel_num,int l,float* pic_h,float* result_h)
{
    //这里的l是不包含自己往左有多少个点

    //进入老流程开始吧咱就
    //首先第一步就是分配空间
    float* pic_d;
    float* result_d;

    cudaMalloc((void**)&pic_d,row*lie*channel_num*sizeof(float));
    cudaMalloc((void**)&result_d,row*lie*channel_num*sizeof(float));

    //接下来是转移数据

    cudaMemcpy(pic_d,pic_h,row*lie*channel_num*sizeof(float),cudaMemcpyHostToDevice);

    //接下来得先计算分配的块和线程的数据

    dim3 blockdata(ceil(lie/16.0),ceil(row/16.0));
    dim3 threaddata(16,16);

    image_smoothing_kernal<<<blockdata,threaddata>>>(row,lie,channel_num,l,pic_d,result_d);

    cudaMemcpy(result_h,result_d,row*lie*channel_num*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(pic_d);
    cudaFree(result_d);
}

//主函数懒得写了ai生成一下啊
int main()
{
    int row, lie, l, channel_num;
    // 输入顺序：行、列、模糊半径、通道数
    if (!(cin >> row >> lie >> l >> channel_num)) return 0;

    float* pic_h = new float[channel_num * row * lie];
    float* result_h = new float[channel_num * row * lie]; // 修正：结果也是彩色的

    // 输入图片数据 (HWC 布局)
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < lie; j++)
        {
            for (int k = 0; k < channel_num; k++) {
                cin >> pic_h[i * lie * channel_num + j * channel_num + k];
            }
        }
    }

    // 调用你写的平滑函数
    image_smoothing(row, lie, channel_num, l, pic_h, result_h);

    // 输出结果 (保持 HWC 格式输出)
    cout << "--- Smoothing Result ---" << endl;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < lie; j++)
        {
            cout << "[ ";
            for (int k = 0; k < channel_num; k++) {
                cout << result_h[i * lie * channel_num + j * channel_num + k] << " ";
            }
            cout << "] ";
        }
        cout << endl;
    }

    delete[] pic_h;
    delete[] result_h;
    return 0;
}