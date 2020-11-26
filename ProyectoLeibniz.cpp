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
proyecto( double *D,  double *F, double *G, int numElements)
{
	#include <math.h>	// Libreria para poder usar pow()
	
    int id = blockDim.x * blockIdx.x + threadIdx.x;
	double temp = 0.f;

    if (id < numElements)
    {
        temp = pow((-1.0), id-1.0);
        double a = 2id-1;
        D[id] = temp/a;
    }
}

/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 5000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host output vector C
    float *h_D = (float *)malloc(size);

    // Verify that allocations succeeded
    if ( h_D == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
// Allocate the device output vector C
    float *d_D = NULL;
    err = cudaMalloc((void **)&d_D, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_D, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
	float suma = 0;
    for (int i =0; i < numElements; ++i)
    {
		suma++;
    }

    printf("Test completed\n");

    

    err = cudaFree(d_D);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
   
    free(h_D);

    
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	printf("La sumatoria converge para 1\n");
    printf("Completed :D\n");
    return 0;
}







"""
#include <stdio.h>
#include <stdlib.h>
main()
{
    double n, i;  // Number of iterations and control variable
    double s = 1; //Signal for the next iteration
    double pi = 0;
    printf("Approximation of the number PI through the Leibniz's series\n");
    printf("\nEnter the number of iterations: ");
    scanf("%lf", &n);
    printf("\nPlease wait. Running...\n");
    for (i = 1; i <= (n * 2); i += 2)
    {
        pi = pi + s * (4 / i);
        s = -s;
    }
    printf("\nAproximated value of PI = %1.16lf\n", pi);
}
"""