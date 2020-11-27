/*----------------------------------------
 * Por Juan Pablo Pineda 19087
 *     Alejandra Gudiel 19232
 *     Oscar Saravia 19322
 *     Julio Herrera 19402
 *     Andres Emilio Quinto 18288
 * ---------------------------------------
 * UNIVERSIDAD DEL VALLE DE GUATEMALA
 * CC3056 - Programacion de Microprocesadores - Sección 10
 * Proyecto 4, segunda serie de potencias:
 * Cálculo del valor de PI mediante la serie matemática de Leibniz y la serie de Nilakantha.
 * Uso de streams individuales y simultáneos para cada serie con arreglos de 1000 elementos.
 *----------------------------------------
*/

#include <stdio.h>
#include <math.h>

// DEVICE

/* Serie Nilakantha
 * Dada por 3 + 4/(2*3*4) - 4/(4*5*6) + 4/(6*7*8) - 4/(8*9*10) + ...
 * Solo se calcula desde el segundo termino que lleva el patrón de denominadores.
 * El "3" inicial se suma en host al sumar todos los demás términos
*/
__global__ void
nilakantha(double *A, int numElements)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < numElements)
    {
        int sign = ((i%2) * 2) - 1; // Da -1 para numeros pares y 1 para numeros impares
        double step = (id + 1.0) * 2.0; // Ya que i empieza en cero, se le suma 1. Indica el primer numero del denomiador de la serie
        double denominator = step * (step+1.0) * (step+2.0); // Calcula el denominador de la serie
        double value = 4.0 / denominator;
		A[id] = value;
		
    }
}

/* Serie Leibniz
 *
*/
__global__ void
leibniz(double *A, int numElements)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < numElements)
    {
        // colocar serie
    }
}

// HOST
int main(void) {
    // Código de error para verificar valores de retorno a llamadas CUDA
    cudaError_t err = cudaSuccess;

    // Numero de iteraciones de la sumatoria "infinita" = numero total de hilos
    int numElements;
    printf("Ingrese el numero de iteraciones de la sumatoria (empieza en n=1): ");
    scanf("%d", &numElements);
    size_t size = numElements * sizeof(double);
    printf("La sumatoria realizara %d iteraciones\n", numElements);

    // Asignando el espacio en memoria para el vector A en host
    double *h_A = (double *)malloc(size);
    // Verificando asignación de memoria en host
    if (h_A == NULL)
    {
        fprintf(stderr, "Fallo al alojar el vector A en el host!\n");
        exit(EXIT_FAILURE);
    }

    // Asignando el espacio en memoria para el vector B en host
    double *h_B = (double *)malloc(size);
    // Verificando asignación de memoria en host
    if (h_B == NULL)
    {
        fprintf(stderr, "Fallo al alojar el vector B en el host!\n");
        exit(EXIT_FAILURE);
    }

    // Asignando el espacio en memoria para el vector A en device
    double *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    // Verifiando asignación de memoria en device
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al alojar el vector A en el device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Asignando el espacio en memoria para el vector B en device
    double *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);
    // Verifiando asignación de memoria en device
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al alojar el vector B en el device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Declarando los 2 streams a usar
    cudaStream_t stream1, stream2;

    // Asignando streams
    // Nota: Otra opción era asignar solo 1 stream con flag "cudaStramNonBlocking" y que la otra serie corrriera en el Null Stream
    err = cudaStreamCreate(&stream1);
    cudaError_t err2 = cudaStreamCreate(&stream2);
    if (err != cudaSuccess || err2 != cudaSuccess)
    {
        fprintf(stderr, "Fallo al crear alguno de los streams (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Creando eventos CUDA para calcular el tiempo de ejecución en los streams
    cudaEvent_t start[2], stop[2];
    cudaEventCreate(&start[0]);
    cudaEventCreate(&stop[0]);
    cudaEventCreate(&start[1]);
    cudaEventCreate(&stop[1]);

    // Calculando bloques e hilos por bloque para ambos streams
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("El kernel CUDA se lanzara con %d bloques de %d hilos cada uno\n", blocksPerGrid, threadsPerBlock);

    // Lanzando el kernel de la serie Nilakantha para el stream 1
    cudaEventRecord(start[0], stream1);
    nilakantha<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, numElements);
    cudaEventRecord(stop[0], stream1);
    /* Este catch de error no sé si lo realiza sobre la llamada a kernel ya que la ejecución es asincrónica
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al realizar la opeeracion kernel 'serie' (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/

    // Copiando el resultado del vector A en la memoria en el device
    // al vector A en la memoria del host, de forma asincrónica
    printf("Copiando la data de salida del device a la memoria del host\n");
    err = cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost, stream1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al copiar el vector A del device al host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Lanzando el kernel de la serie Leibniz para el stream 2
    cudaEventRecord(start[1], stream2);
    leibniz<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_B, numElements);
    cudaEventRecord(stop[1], stream2);
    // Colocar el mismo catch de error que en la llamada a nilakantha, si funciona

    // Copiando el resultado del vector B en la memoria en el device
    // al vector B en la memoria del host, de forma asincrónica
    printf("Copiando la data de salida del device a la memoria del host\n");
    err = cudaMemcpyAsync(h_B, d_B, size, cudaMemcpyDeviceToHost, stream2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al copiar el vector A del device al host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Obteniendo tiempos
    float milliseconds1 = 0, milliseconds2 = 0;
    cudaEventSynchronize(stop[0])
    cudaEventElapsedTime(&milliseconds1, start[0], stop[0]);
    cudaEventSynchronize(stop[0])
    cudaEventElapsedTime(&milliseconds2, start[1], stop[1]);
    printf("Tiempo de ejecución de la serie Nilakantha: %f ms\n", milliseconds1);
    printf("Tiempo de ejecución de la serie Leibniz: %f ms\n", milliseconds2);

    // Destruyendo eventos
    cudaEventDestroy(start[0])
    cudaEventDestroy(stop[0])
    cudaEventDestroy(start[1])
    cudaEventDestroy(stop[2])

    // Destruyendo ambos streams
    //cudaDeviceSynchronize(); No sé si es necesario esperar al device para destruir los streams
    err = cudaStreamDestroy(stream1);
    err2 = cudaStreamDestroy(stream2);
    if (err != cudaSuccess || err2 != cudaSuccess)
    {
        fprintf(stderr, "Fallo al destruir alguno de los streams (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Libera la memoria en el device (ya que ya se copió al host)
    err = cudaFree(d_A);
    err2 = cudaFree(d_B);
    if (err != cudaSuccess || err2 != cudaSuccess)
    {
        fprintf(stderr, "Fallo al liberar el vector A o el vector B en el device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Sumando los resultados para la serie Nilakantha
    double pi = 3.0;
    for (int i = 0; i < numElements; i++)
    {
        pi += h_A[i];
	    //printf("Valor : %.16f\n", h_A[i]);
    }
    printf("Valor aproximado de pi: %.16f\n", pi);
    
    // Sumando los resultados para la serie Leibniz

    // Liberando memoria en el host
    free(h_A);
    free(h_B);

    // Reseteando el device y saliendo
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return 0;

}