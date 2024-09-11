#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

int m;
int n;
int k;
double *A;
double *B;
double *C;

double *DA;
double *DB;
double *DC;

#define BS (16)
#define NDIV (8) // number of division

long time_diff_us(struct timeval st, struct timeval et)
{
    return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

__global__ void matmul_kernel(double *DA, double *DB, double *DC, int m, int n, int k)
{
    int i, j, l;
    int lda = m;
    int ldb = k;
    int ldc = m;
    double cij;

    j = blockIdx.y * blockDim.y + threadIdx.y;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m || j >= n) return;

    cij = DC[i+j*ldc];
    for (l = 0; l < k; l++) {
        cij += DA[i+l*lda] * DB[l+j*ldb];
    }
    DC[i+j*ldc] = cij;
}

/* returns start/end point of the idiv-th division */
/* results are put into *ps and *pe. */
/* rank should be 0 <= idiv = ndiv */
void divide_length(int len, int idiv, int ndiv, int *ps, int *pe)
{
    int llen = (len+ndiv-1)/ndiv; // local length
    int s = llen*idiv;
    int e = llen*(idiv+1);

    if (s > len) s = len;
    if (e > len) e = len;

    *ps = s;
    *pe = e;
    return;
}


int matmul()
{
    int idiv;
    cudaStream_t streams[NDIV];
    int ss[NDIV], es[NDIV];

    // initialize streams
    for (idiv = 0; idiv < NDIV; idiv++) {
        cudaStreamCreate(&streams[idiv]);
        divide_length(n, idiv, NDIV, &ss[idiv], &es[idiv]);
    }

    // copy A from host to device
    cudaMemcpy(DA, A, sizeof(double)*m*k, cudaMemcpyHostToDevice);

    // divide B and C vertically
    // copy host to device
    for (idiv = 0; idiv < NDIV; idiv++) {
        int s = ss[idiv], e = es[idiv];
        // here columns [s, e) are copied, instead of [0, n) columns
        //printf("%d-th div: [%d,%d)\n", idiv, s, e);

        // copy partial B and C
        cudaMemcpyAsync(&DB[k*s], &B[k*s], sizeof(double)*k*(e-s), cudaMemcpyHostToDevice,
                        streams[idiv]);
        cudaMemcpyAsync(&DC[m*s], &C[m*s], sizeof(double)*m*(e-s), cudaMemcpyHostToDevice,
                        streams[idiv]);
    }

    // compute
    for (idiv = 0; idiv < NDIV; idiv++) {
        int s = ss[idiv], e = es[idiv];
        // here columns [s, e) are computed, instead of [0, n) columns

        // invoking (about) m*(e-s) threads
        dim3 grid = dim3((m+BS-1)/BS, (((e-s)+BS-1)/BS), 1);
        dim3 block = dim3(BS, BS, 1);
        matmul_kernel<<<grid, block, 0, streams[idiv]>>>(DA, &DB[k*s], &DC[m*s], m, e-s, k);
    }

    // copy device to host
    for (idiv = 0; idiv < NDIV; idiv++) {
        int s = ss[idiv], e = es[idiv];
        
        // copy partial C from device to host
        cudaMemcpyAsync(&C[m*s], &DC[m*s], sizeof(double)*m*(e-s), cudaMemcpyDeviceToHost,
                        streams[idiv]);
    }

    // Wait for all tasks on all streams are finished
    for (idiv = 0; idiv < NDIV; idiv++) {
        cudaStreamSynchronize(streams[idiv]);
        cudaStreamDestroy(streams[idiv]);
    }

    return 0;
}

int main(int argc, char *argv[])
{
    int i, j;
    int iter;
    cudaError_t rc;

    if (argc < 4) {
        printf("Specify M, N, K.\n");
        exit(1);
    }

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);

    /* allocate matrix region */
#if 1
    rc = cudaHostAlloc(&A, sizeof(double)*m*k, cudaHostAllocMapped);
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc failed\n"); exit(1);
    }
    rc = cudaHostAlloc(&B, sizeof(double)*k*n, cudaHostAllocMapped);
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc failed\n"); exit(1);
    }
    rc = cudaHostAlloc(&C, sizeof(double)*m*n, cudaHostAllocMapped);
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc failed\n"); exit(1);
    }
#else
    A = (double *)malloc(sizeof(double)*m*k);
    B = (double *)malloc(sizeof(double)*k*n);
    C = (double *)malloc(sizeof(double)*m*n);
#endif

    /* setup matrix (column major) */
    /* A is m*k matrix */
    for (j = 0; j < k; j++) {
        for (i = 0; i < m; i++) {
            A[i+j*m] = 1.0;
        }
    }
    /* B is k*n matrix */
    for (j = 0; j < n; j++) {
        for (i = 0; i < k; i++) {
            B[i+j*k] = 10.0;
        }
    }
    /* C is m*n matrix */
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            C[i+j*m] = 0.0;
        }
    }

    /* allocate device memory */
    rc = cudaMalloc((void**)&DA, sizeof(double)*m*k);
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n"); exit(1);
    }
    rc = cudaMalloc((void**)&DB, sizeof(double)*k*n);
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n"); exit(1);
    }
    rc = cudaMalloc((void**)&DC, sizeof(double)*m*n);
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n"); exit(1);
    }

    /* Repeat same computation */
    for (iter = 0; iter < 5; iter++) {
        struct timeval st, et;
        long flop;
        long us;

        gettimeofday(&st, NULL);
        
        /* computation */
        matmul();
        cudaDeviceSynchronize();
        
        gettimeofday(&et, NULL);
        
        flop = 2.0*(double)m*(double)n*(double)k;
        us = time_diff_us(st, et);
        printf("Matmul took %ld us --> %.3lf GFlops  (with data transfer)\n",
               us, (double)flop/(double)us/1000.0);
    }

    cudaFree(DA);
    cudaFree(DB);
    cudaFree(DC);
#if 1
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
#else
    free(A);
    free(B);
    free(C);
#endif
    return 0;
}
