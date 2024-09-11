#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* array size of each data */
#define N 65536
//#define N (64*1024*1024)

/* (1) send data in sendleft to rank-1 */
/* (2) send data in sendright to rank+1 */
/* (3) recv data to recvleft from rank-1 */
/* (4) recv data to recvright from rank+1 */
/* If communication target does not exist (border process), */
/* do nothing */
/* This is unsafe version, which may cause DEADLOCK! */
/* Use neicomm_safe() instead. */
int neicomm_unsafe(int rank, int size,
                   void *sendleft, void *sendright,
                   void *recvleft, void *recvright,
                   int count, MPI_Datatype datatype)
{
    /* (1) */
    if (rank > 0) {
        MPI_Send(sendleft, count, datatype, rank-1, 100, MPI_COMM_WORLD);
    }

    /* (2) */
    if (rank < size-1) {
        MPI_Send(sendright, count, datatype, rank+1, 100, MPI_COMM_WORLD);
    }

    /* (3) */
    if (rank > 0) {
        MPI_Status stat;
        MPI_Recv(recvleft, count, datatype, rank-1, 100, MPI_COMM_WORLD, &stat);
    }

    /* (4) */
    if (rank < size-1) {
        MPI_Status stat;
        MPI_Recv(recvright, count, datatype, rank+1, 100, MPI_COMM_WORLD, &stat);
    }

    return 0;
}

/* Safe version */
int neicomm_safe(int rank, int size,
                 void *sendleft, void *sendright,
                 void *recvleft, void *recvright,
                 int count, MPI_Datatype datatype)
{
    MPI_Request leftreq;
    MPI_Request rightreq;

    /* start (3) */
    if (rank > 0) {
        MPI_Irecv(recvleft, count, datatype, rank-1, 100, MPI_COMM_WORLD, &leftreq);
    }

    /* start (4) */
    if (rank < size-1) {
        MPI_Irecv(recvright, count, datatype, rank+1, 100, MPI_COMM_WORLD, &rightreq);
    }

    /* (1) */
    if (rank > 0) {
        MPI_Send(sendleft, count, datatype, rank-1, 100, MPI_COMM_WORLD);
    }

    /* (2) */
    if (rank < size-1) {
        MPI_Send(sendright, count, datatype, rank+1, 100, MPI_COMM_WORLD);
    }

    /* finsh (3) */
    if (rank > 0) {
        MPI_Status stat;
        MPI_Wait(&leftreq, &stat);
    }

    /* finish (4) */
    if (rank < size-1) {
        MPI_Status stat;
        MPI_Wait(&rightreq, &stat);
    }

    return 0;
}


int main(int argc, char *argv[])
{
    int rank;
    int size;
    int i;
    double *mydata;
    double *leftdata;
    double *rightdata;
    
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    mydata = (double *)malloc(sizeof(double)*N);
    leftdata = (double *)malloc(sizeof(double)*N);
    rightdata = (double *)malloc(sizeof(double)*N);
    
    /* calculate my data */
    for (i = 0; i < N; i++) {
        mydata[i] = (double)rank*(double)rank;
        leftdata[i] = -1.0;
        rightdata[i] = -1.0;
    }
    
    printf("Process %d: before communication. leftdata = %.1lf, mydata = %.1lf, rightdata = %.1lf\n",
           rank, leftdata[0], mydata[0], rightdata[0]);
    
    /* Neighbor communication */
    /* If neicomm_unsafe(...) is called here, this program deadlocks */
#if 1
    printf("Calling neicomm_safe!\n");
    neicomm_safe(rank, size,
                 (void *)mydata, (void *)mydata,
                 (void *)leftdata, (void *)rightdata,
                 N, MPI_DOUBLE);
#else
    printf("Calling neicomm_unsafe!\n");
    neicomm_unsafe(rank, size,
                 (void *)mydata, (void *)mydata,
                 (void *)leftdata, (void *)rightdata,
                 N, MPI_DOUBLE);
#endif
    
    printf("Process %d: after communication, leftdata = %.1lf, mydata = %.1lf, rightdata = %.1lf\n",
           rank, leftdata[0], mydata[0], rightdata[0]);

    free(rightdata);
    free(leftdata);
    free(mydata);
    
    MPI_Finalize();
    return 0;
}
