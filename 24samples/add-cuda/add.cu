#include <stdio.h>
#include <stdlib.h>

#define N (100)
#define BS (5)

__global__ void add(int *DA, int *DB)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    /* printf("Hello GPU world I'm %d\n", id); */
    DA[i] += DB[i];
    return;
}

int main(int argc, char *argv[])
{
    int i;
    int A[N], B[N];
    int *DA, *DB;

    // Initialize arrays
    for (i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i*2;
    }

    cudaMalloc((void **)&DA, sizeof(int)*N);
    cudaMalloc((void **)&DB, sizeof(int)*N);

    cudaMemcpy(DA, A, sizeof(int)*N, cudaMemcpyDefault);
    cudaMemcpy(DB, B, sizeof(int)*N, cudaMemcpyDefault);

    /* call GPU kernel function with N threads */
    add<<<N/BS, BS>>>(DA, DB);

    cudaMemcpy(A, DA, sizeof(int)*N, cudaMemcpyDefault);

    cudaFree(DB);
    cudaFree(DA);

    printf("A[%d]=%d\n", 0, A[0]);
    printf("A[%d]=%d\n", N-1, A[N-1]);

    return 0;
}

