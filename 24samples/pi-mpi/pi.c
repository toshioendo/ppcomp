#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>

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

double pi(int n)
{
    int i;
    int nprocs;
    int rank;
    double lsum = 0.0;
    double sum = 0.0;
    double dx = 1.0 / (double)n;
    int ns, ne;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    //printf("I'm rank %d\n", rank);

    /* [ns, ne) should be computed by this process */
    divide_length(n, rank, nprocs, &ns, &ne);

    for (i = ns; i < ne; i++) {
        double x;
        double y;
        x = (double)i * dx;
        y = sqrt(1.0 - x*x);

        lsum += dx*y;
    }
    
    /* Accumulate "lsum" of all processes to "sum" of rank 0 */
    MPI_Reduce(&lsum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Only rank 0 returns the correct answer */
    return 4.0*sum;
}

int main(int argc, char *argv[])
{
    int nsamples;
    int i;
    int rank;
    int size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        printf("Number of processes is %d\n", size);
    }

    if (argc < 2) {
        if (rank == 0) {
            printf("Specify #samples.\n");
        }
        MPI_Finalize();
        exit(1);
    }

    nsamples = atoi(argv[1]);


    /* Repeat same computation for 3 times */
    for (i = 0; i < 3; i++) {
        struct timeval st;
        struct timeval et;
        long us;
        double res;

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&st, NULL); /* get start time */

        res = pi(nsamples);

        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&et, NULL); /* get end time */

        if (rank == 0) {
            /* Rank 0 prints the answer */
            us = time_diff_us(st, et);
            printf("Result=%.15lf: Pi took %ld us --> %lf Msamples/sec\n",
                   res, us, (double)nsamples/(double)us);
        }
    }

    MPI_Finalize();
    return 0;
}
