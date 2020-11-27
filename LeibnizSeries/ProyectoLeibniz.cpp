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
leibniz(double *C,int numElements)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double a = 1;
    //denominador
    double n;
    if (i < numElements)
    {
        a = A[i]*(pow(-1.f, i + 1)/(2*i - 1));
        B[i] = 4.f/a;
        C[i] = a/n;
		printf("El resultado es: %20.18f\n ",C[i]);

    }
    """
    idea: hacer que el numerador de la serie se guarde en el vector A como 4^2*i+1
    y en el vector B guardar la parte del denominador y en el C hacer un for A / B
    """

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
    size_t size = numElements * sizeof(double);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host output vector C
    double *h_C = (double *)malloc(size);

    // Verify that allocations succeeded
    if ( h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
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

    

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    leibniz<<<blocksPerGrid, threadsPerBlock>>>(d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch leibniz kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
	double suma = 0;
    for (int i =0; i < numElements; ++i)
    {
		suma++;
    }

    printf("Test completed\n");

    

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
   
    free(h_C);

    
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Completed :D\n");
    return 0;
}