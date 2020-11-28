/*----------
* Streams - simple multi-stream example
* GPU Pro Tip: CUDA 7 Streams simplify concurrency
* NVIDIA Developer Blog
* Autor: Mark Harris
* ----------
* Universidad del Valle
* Programación de Microprocesadores
* Mod.: K.Barrera, J.Celada
* Semestre 2 2020
* ----------
*/


#include <stdio.h>
#include <math.h>

//(const int N = 1 << 20;
const int N = 100;

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main()
{
	const int num_streams = 2;
    cudaStream_t streams[num_streams];
    float *h_data[num_streams], *d_data[num_streams];
	
	
    for (int i = 0; i < num_streams; i++) 
	{
		
        cudaStreamCreate(&streams[i]);
		
		h_data[i] = (float *)malloc(N*sizeof(float)); 
        cudaMalloc(&d_data[i], N * sizeof(float));
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(d_data[i], N);
		cudaMemcpyAsync(h_data[i], d_data[i], N*sizeof(float),cudaMemcpyDeviceToHost, streams[i]);
		

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
    }
	
	
	for(int i = 0; i < N; i++)
		printf("Value %d is: %f\n", i, h_data[0][i]);
	
    cudaDeviceReset();

    return 0;
}