#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>
#include <cuda_runtime.h>

int m;
int n; /* The number of columns of global B/C */
int ln; /* The number of columns of process local B/C */
int k;
double *A;
double *LB;
double *LC;

double *DA;
double *DLB;
double *DLC;

#define BS (16)

long time_diff_us(struct timeval st, struct timeval et)
{
    return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}


/* returns start/end point of the rank-th process */
/* results are put into *ps and *pe. */
/* rank should be 0 <= rank = nprocs */
void divide_length(int len, int rank, int nprocs, int *ps, int *pe)
{
    int llen = (len+nprocs-1)/nprocs;
    int s = llen*rank;
    int e = llen*(rank+1);

    if (s > len) s = len;
    if (e > len) e = len;

    *ps = s;
    *pe = e;
    return;
}

__global__ void matmul_kernel(double *DA, double *DLB, double *DLC, int m, int ln, int k)
{
    int i, j, l;
    int lda = m;
    int ldb = k;
    int ldc = m;
    double cij;

    j = blockIdx.y * blockDim.y + threadIdx.y;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m || j >= ln) return;

    cij = DLC[i+j*ldc];
    for (l = 0; l < k; l++) {
        cij += DA[i+l*lda] * DLB[l+j*ldb];
    }
    DLC[i+j*ldc] = cij;
}

int matmul()
{
    /* invoking (about) m*ln threads */
    dim3 grid = dim3((m+BS-1)/BS, ((ln+BS-1)/BS), 1);
    dim3 block = dim3(BS, BS, 1);
    matmul_kernel<<<grid, block>>>(DA, DLB, DLC, m, ln, k);

    return 0;
}

int main(int argc, char *argv[])
{
    int i, j;
    int rank, nprocs;
    char hostname[64];
    cudaError_t rc;

    MPI_Init(&argc, &argv);

    if (argc < 4) {
        printf("Specify M, N, K.\n");
        MPI_Finalize();
        exit(1);
    }

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);

    /* get the rank of this process */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    gethostname(hostname, 63);

    {
        int ndev;
        cudaGetDeviceCount(&ndev);

        rc = cudaSetDevice(rank % ndev);
        if (rc != cudaSuccess) {
            printf("rank %d: cudaSetDevice(%d) FAILED\n", rank, ndev);
            exit(1);
        }

        printf("rank %d/%d@%s: I'll use device %d/%d\n", rank, nprocs, hostname, rank%ndev, ndev);
    }

    {
        int s, e;
        /* calculates the number of columns of process local B/C */
        divide_length(n, rank, nprocs, &s, &e);
        printf("rank %d takes [%d,%d)\n", rank, s, e);
        ln = e-s;
    }

    /* allocate matrix region */
    A = (double *)malloc(sizeof(double)*m*k);
    if (ln > 0) {
        LB = (double *)malloc(sizeof(double)*k*ln);
        LC = (double *)malloc(sizeof(double)*m*ln);
    }
    else {
        LB = NULL;
        LC = NULL;
    }

    /* setup matrix (column major) */
    /* A is m*k matrix */
    for (j = 0; j < k; j++) {
        for (i = 0; i < m; i++) {
            A[i+j*m] = 1.0;
        }
    }
    /* LB is k*ln matrix */
    for (j = 0; j < ln; j++) {
        for (i = 0; i < k; i++) {
            LB[i+j*k] = 10.0;
        }
    }
    /* LC is m*ln matrix */
    for (j = 0; j < ln; j++) {
        for (i = 0; i < m; i++) {
            LC[i+j*k] = 0.0;
        }
    }

    /* allocate device memory */
    rc = cudaMalloc((void**)&DA, sizeof(double)*m*k);
    if (rc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n"); exit(1);
    }
    if (ln > 0) {
        rc = cudaMalloc((void**)&DLB, sizeof(double)*k*ln);
        if (rc != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed\n"); exit(1);
        }
        rc = cudaMalloc((void**)&DLC, sizeof(double)*m*ln);
        if (rc != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed\n"); exit(1);
        }
    }
    else {
        DLB = NULL;
        DLC = NULL;
    }

    /* Repeat same computation for 5 times */
    for (i = 0; i < 5; i++) {
        struct timeval st;
        struct timeval et;
        long us;

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&st, NULL); /* get start time */

        if (ln > 0) {
            /* copy input matrices from host to device */
            cudaMemcpy(DA, A, sizeof(double)*m*k, cudaMemcpyHostToDevice);
            cudaMemcpy(DLB, LB, sizeof(double)*k*ln, cudaMemcpyHostToDevice);
            cudaMemcpy(DLC, LC, sizeof(double)*m*ln, cudaMemcpyHostToDevice);

            /* computation */
            matmul();

            /* copy output matrices from device to host */
            cudaMemcpy(LC, DLC, sizeof(double)*m*ln, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&et, NULL); /* get start time */

        if (rank == 0) {
            us = time_diff_us(st, et);
            printf("Matmul took %ld us --> %.3lf GFlops\n",
                   us, 2.0*(double)m*(double)n*(double)k/(double)us/1000.0);
        }
    }

    cudaFree(DA);
    if (ln > 0) {
        cudaFree(DLB);
        cudaFree(DLC);
    }

    free(A);
    if (ln > 0) {
        free(LB);
        free(LC);
    }

    MPI_Finalize();
    return 0;
}
