// ********************************************************************************************************************
// PURPOSE      :   Index calculations for 2D Grid block with 2D thread block (2X2)                                   *
// LANGUAGE     :                 CUDA C / CUDA C++																      *
// ASSUMPTIONS  :   2D Configuration 4 threads in each x & y directions with thread block of (2X2)                    *
// DATE         :   1 April 2020                                                                                      *
// AUTHOR       :   Vaibhav BENDRE                                                                                    *
//                  vaibhav.bendre7520@gmail.com                                                                      *
// ********************************************************************************************************************

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//------------------------------------------------------
//               C++ HEADR FILES                        |
//------------------------------------------------------

#include<stdio.h>
#include<stdlib.h>
#include<iostream>


// Kernel/Device code for CUDA GPU - unique index calculations for accessing array elements

__global__ void uniqueIndexCalc2D_2DConfig2(int* arr){
	
	unsigned int blockOffset{ blockDim.x * blockDim.y * blockIdx.x };
	unsigned int rowOffset{(blockDim.x * blockDim.y * gridDim.x)* blockIdx.y};
	unsigned int threadIndex {(threadIdx.y * blockDim.x ) + threadIdx.x};
	unsigned int globalIndex { blockOffset + rowOffset + threadIndex };
	
	printf("blockOffset : %d  rowOffset : %d  threadIndex : %d  globalIndex : %d  value : %d \n",
			blockOffset, rowOffset, threadIndex, globalIndex, arr[globalIndex]);
}


// HOST code for CPU

int main(){

	unsigned int Nx{4}, Ny[4];
	int arrSize{16};
	int arrMemorySize{ static_cast<int>(sizeof(int)) * arrSize };
	
	// Initialize the data in CPU context
	int arrData[]{ 12,23,45,21,57,87,89,34,57,68,86,58,15,75,97,69 };
	
	// Print array data 
	for( unsigned int iCounter{0}; iCounter < arrSize; ++iCounter ){
		std::cout << arrData[iCounter] << "   ";
	}
	std::cout << "\n\n\n\n";
	
	int* data;
	
	cudaMalloc((void**)&data, arrMemorySize);
	
	// Transfer context from CPU to GPU
	cudaMemcpy(data, arrData, arrMemorySize, cudaMemcpyHostToDevice);
	
	dim3 block(2,2,1);
	dim3 grid(Nx/block.x , Ny/ block.y);

	// Launch Kernel with specified grid and block 
	uniqueIndexCalc2D_2DConfig2 <<< grid, block >>>(data);
	
	cudaDeviceSynchronize();
	
		// reclaim CPU and GPU memory
	cudaDeviceReset();
	
	return 0;
	
}