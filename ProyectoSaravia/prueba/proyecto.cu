/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
nilakantha(double *A, int numElements)
{
	#include <math.h>	// Libreria para poder usar pow()
	
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < numElements)
    {
        double step = (id + 1.0) * 2.0;
        double denominator = step * (step+1.0) * (step+2.0);
        double value = 4.0 / denominator;
		A[id] = value;
		
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Numero de iteraciones de la sumatoria "infinita" = numero total de hilos
    int numElements;
    printf("Ingrese el numero de iteraciones de la sumatoria (empieza en n=1): ");
    scanf("%d", &numElements);
    size_t size = numElements * sizeof(double);
    printf("La sumatoria realizara %d iteraciones\n", numElements);

    // Allocate the host input vector A
    double *h_A = (double *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
	/*
    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
		// Llenar los vectores
		h_B[i] = 0;
		h_C[i] = 0;
		
    }
	h_C[0] = 2;
	h_C[1] = 3;
	h_C[2] = 4;
	for (int i = 0; i < numElements; ++i)
    {
		h_A[i] = h_C[0]*h_C[1]*h_C[2];
		h_C[0] = h_C[0] + 2;
		h_C[1] = h_C[1] + 2;
		h_C[2] = h_C[2] + 2;
		
    }*/

    // Allocate the device input vector A
    double *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    /*printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    nilakantha<<<blocksPerGrid, threadsPerBlock>>>(d_A, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch proyecto kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    double pi = 3.0;
    for (int i = 0; i < numElements; i++)
    {
		if (i%2 == 0) {
            pi += h_A[i];
        } else {
            pi -= h_A[i];
        }
	    printf("Valor : %.16f\n", h_A[i]);
		
    }
	printf("Valor aproximado de pi: %.16f\n", pi);
	//printf("Se puede observar que el valor obtenido en los diferentes valores de n de la sumatoria va en aumento y no tiende a un valor fijo, por lo tanto la serie diverge.");

    //printf("Test PASSED\n");

    // Free device global memory

    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

