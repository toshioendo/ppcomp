#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include <assert.h>
#include <malloc.h>

#define BUFSIZE (64*1024*1024)

char *sbuf;
char *rbuf;

int rank;
int size;


int main(int argc, char *argv[])
{
    char hostname[64];
    int msgsize;
    int tag = 1;
    MPI_Init(&argc, &argv);

#if 1
    sbuf = malloc(BUFSIZE);
    rbuf = malloc(BUFSIZE);
#else
    sbuf = valloc(BUFSIZE);
    rbuf = valloc(BUFSIZE);
#endif

    gethostname(hostname, 63);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank < 4) {
	printf("%s:%d rank=%d, size=%d. sbuf=%p, rbuf=%p\n", 
	       hostname, getpid(), rank, size, sbuf, rbuf);
	fflush(stdout);
    }
    sleep(1);

    {
       memset(sbuf, (unsigned char)rank, BUFSIZE);
       memset(rbuf, (unsigned char)rank, BUFSIZE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (msgsize = 1; msgsize <= BUFSIZE; msgsize *= 2) {
       int iter;
       if (rank == 0) {
         printf("#proc=%d\n", size);
       }
       for (iter = 0; iter < 5; iter++) {
	 double us;
	 double st, et;
	 int buddy = rank ^ 1;
	 int tag = 100+iter;
	 MPI_Status stat;
	 MPI_Barrier(MPI_COMM_WORLD);
	 st = MPI_Wtime();
	 /* MPI_Bcast */
	 MPI_Bcast(rbuf, msgsize, MPI_BYTE, 0, MPI_COMM_WORLD);
	 MPI_Barrier(MPI_COMM_WORLD);

	 et = MPI_Wtime();
	 us = (et-st)*1000000;
	 if (rank == 0) {
	   double speed = (double)msgsize/us;
	   printf("BCAST:     %d Bytes --> %.0lf us, %.1lf MB/s\n",
		   msgsize, us, speed);
	   fflush(0);
	 }

	 /* Send&Recv (flat-tree) */
	 MPI_Barrier(MPI_COMM_WORLD);
	 st = MPI_Wtime();
	 if (rank == 0) {
	   int dst;
	   for (dst = 1; dst < size; dst++) {
	     MPI_Send(sbuf, msgsize, MPI_BYTE, dst, 100, MPI_COMM_WORLD);
	   }
	 }
	 else {
	   MPI_Status stat;
	   MPI_Recv(rbuf, msgsize, MPI_BYTE, 0, 100, MPI_COMM_WORLD, &stat);
	 }

	 MPI_Barrier(MPI_COMM_WORLD);
	 et = MPI_Wtime();
	 us = (et-st)*1000000;
	 if (rank == 0) {
	   double speed = (double)msgsize/us;
	   printf("Send&Recv: %d Bytes --> %.0lf us, %.1lf MB/s\n",
		   msgsize, us, speed);
	   fflush(0);
	 }
       }

       if (rank == 0) {
	 printf("\n");
	 fflush(0);
       }
    }

    MPI_Finalize();
}
