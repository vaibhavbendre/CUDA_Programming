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
// ISSUE :- In this configuration obly first 4 values be printed for all 8 threads		|
// HOW TO FIX THE ISSUE :-																|
//			In order to access the correct element value in an array we should provide	|
//	offset to each thread block.														|
//																						|
//			globalID = threadID + OFFSET												|
//			globalID = threadID + (blockIdx.x * blockDim.x)								|
//																						|
// We can exapand this to 3D configuration as per the need.								|
// NOTE :- Above is just an example and implementation uses only the concept			|
//--------------------------------------------------------------------------------------


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<stdlib.h>
#include<iostream>


//__global__ void uniqueIdxCalcUsingThreadIdx(int* arr) {
//	unsigned int tidx{ threadIdx.x };
//	printf("threadIdx : %d,   value : %d \n", tidx, arr[tidx]);
//}


// This is new kernel function written to fix our issue.
__global__ void uniqueGlobalIdCalcu(int* arr) {

	unsigned int threadIDX{ threadIdx.x };
	unsigned int globalIDX{ threadIDX + (blockIdx.x * blockDim.x) };
	printf("threadIdx : %d.  blockIdx : %d,  blockDim : %d,  value : %d \n", 
		threadIDX,blockIdx.x,blockDim.x, arr[globalIDX]);

}

int main() {

	int arrSize{ 16 };
	int arrByteSize{ static_cast<int>(sizeof(int))* arrSize };
	int arrData[]{ 213,91,14,52,59,28,51,13,97,57,73,52,42,56,44,99 };

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

	uniqueGlobalIdCalcu <<< grid, block >>> (data);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}