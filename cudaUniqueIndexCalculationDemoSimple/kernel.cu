// **********************************************************************************************************
// PURPOSE		:	Index calculation for array to access data from kernel. 								*
// LANGUAGE		:		CUDA C / CUDA C++																	*
// ASSUMPTIONS	:	This code is part of CUDA fundamentals section.		 									*
// DATE			:	24 March 2020																			*
// AUTHOR		:	Vaibhav BENDRE (vaibhav.bendre7520@gmail.com) 											*
//																											*
// **********************************************************************************************************
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<stdlib.h>
#include<iostream>


__global__ void uniqueIdxCalcUsingThreadIdx(int* arr) {
	unsigned int tidx{ threadIdx.x };
	printf("threadIdx : %d,   value : %d \n", tidx, arr[tidx]);
}

int main() {

	int arrSize{ 16 };
	int arrByteSize{ static_cast<int>(sizeof(int))* arrSize };
	int arrData[]{ 23,9,4,53,65,12,1,33,87,45,23,12,342,56,44,99 };

	for ( int iCounter{ 0 }; iCounter < arrSize; ++iCounter) {
		
		std::cout << arrData[iCounter] << "   ";

	}
	std::cout << "\n\n\n";

	int* data;

	cudaMalloc((void**)&data, arrByteSize);
	cudaMemcpy(data, arrData, arrByteSize, cudaMemcpyHostToDevice);

	dim3 block{ 4 };
	dim3 grid{ 4 };

	uniqueIdxCalcUsingThreadIdx <<< grid, block >>> (data);
	
	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}