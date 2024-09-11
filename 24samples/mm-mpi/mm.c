#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

int m;
int n; /* The number of columns of global B/C */
int ln; /* The number of columns of process local B/C */
int k;
double *A;
double *LB;
double *LC;

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

int matmul()
{
    int i, j, l;
    int lda = m;
    int ldb = k;
    int ldc = m;

    for (j = 0; j < ln; j++) {
        for (l = 0; l < k; l++) {
            double blj = LB[l+j*ldb];
            for (i = 0; i < m; i++) {
                double ail = A[i+l*lda];
                LC[i+j*ldc] += ail*blj;
            }
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    int i, j;
    int rank, nprocs;

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

    /* Repeat same computation for 5 times */
    for (i = 0; i < 5; i++) {
        struct timeval st;
        struct timeval et;
        long us;

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&st, NULL); /* get start time */

        matmul();

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&et, NULL); /* get start time */

        if (rank == 0) {
            us = time_diff_us(st, et);
            printf("Matmul took %ld us --> %.3lf GFlops\n",
                   us, 2.0*(double)m*(double)n*(double)k/(double)us/1000.0);
        }
    }

    free(A);
    free(LB);
    free(LC);

    MPI_Finalize();
    return 0;
}
