// ********************************************************************************************************************
// PURPOSE      :   Index calculations for 2D Grid block with 1D thread block                                         *
// LANGUAGE     :                 CUDA C / CUDA C++																      *
// ASSUMPTIONS  :   2D Configuration 8 threads in each x & y directions with thread block of (2X2)                    *
// DATE         :   31 March 2020                                                                                     *
// AUTHOR       :   Vaibhav BENDRE                                                                                    *
//                  vaibhav.bendre7520@gmail.com                                                                      *
// ********************************************************************************************************************

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<stdlib.h>
#include<iostream>

__global__ void uniqueIndex2DCalc(int* arr){

	//GLOBAL INDEX  = ROW OFFSET + COLUMN OFFSET + THREAD ID
	//ROW OFFSET    = gridDim.x * blockDim.x * blockIdx.y
	//COLUMN OFFSTE = blockDim.x * blockIdx.x 
	unsigned int globalIndex{ (gridDim.x * blockDim.x * blockIdx.y) + (blockDim.x * blockIdx.x) + threadIdx.x };
	printf("threadIdx : %d  blockIdx : %d  globalIndex : %d  value : %d \n",
			threadIdx.x , blockIdx.x , globalIndex, arr[globalIndex]);
}


int main(){
	
	int arrSize{16};
	int arrMemorySize{ static_cast<int>(sizeof(int)) * arrSize };
	int arrData[] {12,34,23,12,34,54,123,45,67,87,98,34,25,16,35,87};
	
	for(int iCounter{0}; iCounter < arrSize; ++iCounter){
		std::cout << arrData[iCounter] << "  ";
	}
	std::cout << "\n\n\n\n";
	
	int* data;
	
	//CUDA Dynamic Memory Allocation
	//CUDA Malloc Syntax :
	//	cudaMalloc(void** devPtr, size_t size);
	
	//Parameters:
	//      devPtr    -   pointer to allocated device memory
	//      size      -   size of memory to be copied in bytes
	
	cudaMalloc((void**)&data, arrMemorySize);
	
	//CUDA MemCpy 
	//    cudaMemcpy(void* dst , const void* src, size_t size, enum cudaMemcpyKind);
	
	//Parameters:
	//      dst      -    Destination memory address
	//      src      -    Source memory address
	//      size     -    size in bytes to copy
	//      kind     -    Type of data transfer
	
	cudaMemcpy(data,arrData, arrMemorySize, cudaMemcpyHostToDevice);
	
	//Grid configuration setup - 2D Grid with 1D threadBlock
	
	unsigned int numThreadsXDim{8}, numThreadsYDim{2};
	
	dim3 block(4,1);
	dim3 grid( numThreadsXDim/block.x , numThreadsYDim/block.y );
	
	uniqueIndex2DCalc<<<grid, block>>>(data);
	
	cudaDeviceSynchronize();
	cudaDeviceReset();
	
	return 0;
}
