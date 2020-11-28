/*
*----------------------------------------
* Por Juan Pablo Pineda 19087
*     Alejandra Gudiel 19232
*     Oscar Saravia 19322
*     Julio Herrera 
*     Andres Emilio Quinto 18288
* ---------------------------------------
* UNIVERSIDAD DEL VALLE DE GUATEMALA
* CC3056 - Programacion de Microprocesadores
* Proyecto numero 4, segunda serie de potencias.
*----------------------------------------*/

#include <stdio.h>
#include <cuda_runtime.h>
__global__ void
leibniz(double *A, double* B, double *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        for (int i = 0; i < numElements; i++)
        {
        A[i] = powf(-4, i-1);
        B[i] = ((2*i)-1);
        C[i] = A[i]/B[i];
		printf("El resultado es: %20.18f\n ",C[i]);		
        }
    }
    /**
    idea: hacer que el numerador de la serie se guarde en el vector A como -4^2*i+1
    y en el vector B guardar la parte del denominador y en el C hacer un for A / B
    **/
}

/**
 * Host main routine
 */
/**
 * Host main routine
 */
int
main(void)
{
    #include<math.h>
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 10000;
    size_t size = numElements * sizeof(double);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    double *h_A = (double *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
	

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
		
    }


	


    // Allocate the device input vector A
    double *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    double *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    double *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    leibniz<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch proyecto kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
	h_B[0] = 0.1666666667f;
	float pi = 3.f;
    for (int i = 1; i < numElements; ++i)
    {
		
		pi += (h_B[i-1] + h_B[i]);
		//printf("pi: %f\n", pi);
		
    }
	printf("Valor aproximado de pi: %f", pi);
	//printf("Se puede observar que el valor obtenido en los diferentes valores de n de la sumatoria va en aumento y no tiende a un valor fijo, por lo tanto la serie diverge.");

    //printf("Test PASSED\n");

    // Free device global memory

    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

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