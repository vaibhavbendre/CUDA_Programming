// ******************************************************************************************
// PURPOSE		:	First programm in CUDA (Hello World)									*
// LANGUAGE		:		CUDA C / CUDA C++													*
// ASSUMPTIONS	:	It is assumed that programmer has no experience in the 					*
//					CUDA as such.															*
// DATE			:	23 March 2020															*
// AUTHOR		:	Vaibhav BENDRE 															*
//					vaibhav.bendre7520@gmail.com											*
// ******************************************************************************************


//--------------------------------------------------
//			CUDA HEADER LIBRARIES					|
//--------------------------------------------------
#include "cuda_runtime.h"			// The C++ API with CUDA specific wrapper that deals with symbols, textures and device functions.
#include "device_launch_parameters.h" // Enables kernel launching parameters for device


//		C or C++ HEADER LIBRARIES
#include<stdio.h>


//Kernel launch code
__global__ void hello_world() {

	printf("Hello CUDA world \n");

}


int main() {

	// This is async kernel call. User needs to explicitly make host function wait till kernel execution.
	hello_world << < 1, 4 >> > ();

	cudaDeviceSynchronize();	// To make host function to wait and sync with kernel.

	cudaDeviceReset();	// Reset the deveice.
	return 0;
}