// ******************************************************************************************************
// PURPOSE		:	Print thread IDs for the 256 threads of a 2D configuration (16 * 16)				*
// LANGUAGE		:		CUDA C / CUDA C++																*
// ASSUMPTIONS	:	2D Configuration 16 threads in each x & y directions with thread block of (8*8)		*
//					threadIdx.z value will be zero since it is 2D configuration							*
// DATE			:	23 March 2020																		*
// AUTHOR		:	Vaibhav BENDRE 																		*
//					vaibhav.bendre7520@gmail.com														*
// ******************************************************************************************************

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>

__global__ void printThreadIDs() {

	printf("\n threadIdx.x : %d,   threadIdx.y :  %d ",threadIdx.x,threadIdx.y);

}

int main() {

	unsigned int Nx{ 16 }, Ny{ 16 }; // multiplication yields to 256 threads in x & y

	dim3 block(8, 8);
	dim3 grid(Nx / block.x, Nx / block.y);

	printThreadIDs <<<grid, block >> > ();
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}