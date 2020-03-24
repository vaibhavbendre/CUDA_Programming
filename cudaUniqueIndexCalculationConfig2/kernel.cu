// **********************************************************************************************************
// PURPOSE		:	Index calculation for array to access data from kernel. 								*
// LANGUAGE		:		CUDA C / CUDA C++																	*
// ASSUMPTIONS	:	This code is part of CUDA fundamentals section.		 									*
// DATE			:	24 March 2020																			*
// AUTHOR		:	Vaibhav BENDRE (vaibhav.bendre7520@gmail.com) 											*
//																											*
// **********************************************************************************************************

//--------------------------------------------------------------------------------------
//			CONFIGURATION 2 - 1D Grid with 2 Thread Blocks								|
// Array			| 23 | 9 | 4 | 53 | 65 | 12 | 1 | 33 |								|
//																						|
// threadIdx.x	->    0	   1   2   3        0    1    2   3								|
//																						|
// Grid				| A  | B | C | D | -- | E |  F  | G | H |							|
//																						|
// In this configuration obly first 4 values be printed for all 8 threads				|
// NOTE :- Above is just an example and implementation uses only the concept			|
//--------------------------------------------------------------------------------------


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

	for (int iCounter{ 0 }; iCounter < arrSize; ++iCounter) {

		std::cout << arrData[iCounter] << "   ";

	}
	std::cout << "\n\n\n";

	int* data;

	cudaMalloc((void**)&data, arrByteSize);
	cudaMemcpy(data, arrData, arrByteSize, cudaMemcpyHostToDevice);

	// These variable we need to modify from code submitted as a simple code 
	dim3 block{ 8 };	// thread block of 8 grids each
	dim3 grid{ 2 };		// for 16 threads we will have 2 grid blocks

	uniqueIdxCalcUsingThreadIdx << < grid, block >> > (data);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}