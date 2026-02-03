//当前作业是rgb图转灰色
#include<iostream>

using namespace std;

__global__
void grayscale_kernal(float* pic_d,float* result_pic_d,int channel_numbers,int row,int lie)
{
    /*
    第一个参数是原始图片的指针
    第二个参数是目标图片的指针
    第三个参数是通道的数目
    第四个参数是行数
    第五个参数是列数
    */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x>=lie || y>=row)return;

    float res = 0;
    int index = y * lie + x;
    for(int i=0;i<channel_numbers;i++) //通道数值加和的方法不是我么的呢重点这里我们假设就是直接各通道直接求均值。
    {
        res += pic_d[index * channel_numbers + i];
    }
    result_pic_d[index] = res/channel_numbers;

    //至此核函数搞定芜湖，fuckyouman！！！！
}

//我们这里不预设一定是rgb这种三维度的图片我们假设他是可能有很多channel
void grayscale(int channel_numbers,float* pic,int row,int lie,float* result_pic)
{
    //这里图片是row x lie 的图片
    //然后这张图的存储方式是每channel个地址存储的是一个点的每个channel的数值。
    //所以开始正常走流程，首先是分配空间
    float* pic_d;
    float* result_pic_d;
    cudaMalloc((void**)&pic_d,channel_numbers*row*lie*sizeof(float));
    cudaMalloc((void**)&result_pic_d,row*lie*sizeof(float));

    //接下来是拷贝数据到gpu上面
    cudaMemcpy(pic_d,pic,channel_numbers*row*lie*sizeof(float),cudaMemcpyHostToDevice);

    //接下来计算网格里面的块数和每个块里面的线程
    dim3 getblockdim(ceil(lie/16.0),ceil(row/16.0));
    dim3 getthreaddim(16,16);

    //接下来是调用核函数
    grayscale_kernal<<<getblockdim,getthreaddim>>>(pic_d,result_pic_d,channel_numbers,row,lie);
    /*
    第一个参数：原始图片的指针
    第二个参数：灰色图片的指针
    第三个参数：channel的数目
    第四个参数：row
    第五个参数：lie
    */

    //现在核函数已经装完逼了现在把结果拷贝回主机
    cudaMemcpy(result_pic,result_pic_d,row*lie*sizeof(float),cudaMemcpyDeviceToHost);

    //还得释放空间
    cudaFree(pic_d);
    cudaFree(result_pic_d);
}


int main()
{
    int channelnum,row,lie;
    cin>>channelnum>>row>>lie;
    float *pic_h = new float[channelnum*row*lie];
    float *result = new float[row*lie];
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<lie;j++)
        {
            for(int k=0;k<channelnum;k++)cin>>pic_h[i*lie*channelnum+j*channelnum+k];
        }
    } 
    grayscale(channelnum,pic_h,row,lie,result);
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<lie;j++)cout<<result[i*lie+j]<<" ";
        cout<<endl;
    }
    return 0;
}