#include<iostream>

using namespace std;

int main()
{
    int num;
    cudaGetDeviceCount(&num);

    cout<<num<<endl;
    
}