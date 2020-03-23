// ******************************************************************************************************
// PURPOSE		:	Print values for CUDA runtime variables for 3D configuration (4*4*4) threads.		*
// LANGUAGE		:		CUDA C / CUDA C++																*
// ASSUMPTIONS	:	3D Configuration 64 threads in each x,y & directions with thread block of (2*2*2)	*
// DATE			:	23 March 2020																		*
// AUTHOR		:	Vaibhav BENDRE 																		*
//					vaibhav.bendre7520@gmail.com														*
// ******************************************************************************************************

#include "cuda_runtime.h"					// The C++ API with CUDA specific wrapper that deals with symbols, textures and device functions.
#include "device_launch_parameters.h"		// Enables kernel launching parameters for device

#include<stdio.h>

__global__ void displayAttributeValues() {

	printf("\nthreadIdx.x : %d,  threadIdx.y : %d,  threadIdx.z : %d,"
		"  blockIdx.x : %d,  blockIdx.y : %d,  blockIdx.z : %d,"
		"  blockDim.x : %d,  blockDim.y : %d,  blockDim.z : %d,"
		"  gridDim.x : %d,  gridDim.y : %d,  gridDim.z : %d\n",
		threadIdx.x,threadIdx.y,threadIdx.z,
		blockIdx.x, blockIdx.y, blockIdx.z,
		blockDim.x, blockDim.y, blockDim.z,
		gridDim.x, gridDim.y, gridDim.z);

}

int main() {
	//There are in total 64 threads that we need to arrange in a desired configuration.

	unsigned int Nx{ 4 }, Ny{ 4 }, Nz{ 4 };

	// We need to yield to following kernel call syntax 
	// kernelName <<< numberOfBlocks, threadsPerBlock >>>()

	dim3 block(2, 2, 2); // This refers to 1 thread block made up how many threads in x,y,z.
	dim3 grid(Nx / block.x, Ny / block.y, Nz / block.z);
	
	//This kernel call is async call
	displayAttributeValues <<< grid, block >>> ();

	// Synchronize the call ask the kernel to wait for host function to complete the execution
	cudaDeviceSynchronize();  
	cudaDeviceReset();
	return 0;

}