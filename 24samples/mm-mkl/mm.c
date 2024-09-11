#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>

int m;
int n;
int k;
double *A;
double *B;
double *C;

long time_diff_us(struct timeval st, struct timeval et)
{
    return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

int matmul()
{
    int i, j, l;
    int lda = m;
    int ldb = k;
    int ldc = m;
    
    char ta = 'N', tb = 'N';
    double alpha = 1.0;
    double beta = 1.0;

    // calls a matrix multiply API of blas library
    dgemm(&ta, &tb, &m, &n, &k, 
          &alpha, A, &lda, B, &ldb,
          &beta, C, &ldc);

    return 0;
}

int main(int argc, char *argv[])
{
    int i, j;

    if (argc < 4) {
        printf("Specify M, N, K.\n");
        exit(1);
    }

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);

    /* allocate matrix region */
    A = (double *)malloc(sizeof(double)*m*k);
    B = (double *)malloc(sizeof(double)*k*n);
    C = (double *)malloc(sizeof(double)*m*n);

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

    /* Repeat same computation for 5 times */
    for (i = 0; i < 5; i++) {
        struct timeval st;
        struct timeval et;
        long us;

        gettimeofday(&st, NULL); /* get start time */
        matmul();
        gettimeofday(&et, NULL); /* get start time */

        us = time_diff_us(st, et);
        printf("Matmul took %ld us --> %.3lf GFlops\n",
               us, 2.0*(double)m*(double)n*(double)k/(double)us/1000.0);
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
