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
leibniz( double *D,  long int n, int numElements){
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    double ans = 0;
    int step = blockDim.x*gridDim.x;
    for (long int i=tid; i < n; i+=step)
        ans = ans + 4.0/(2.0*i+1.0);
    if (tid%2==1)
        ans = -ans;
    result [tid] = ans;
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