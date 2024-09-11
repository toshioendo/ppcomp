#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank;
    int size;
    char hostname[64];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(hostname, 63);
    printf("I'm %d-th process out of %d processes. I'm on node %s\n", 
           rank, size, hostname);

    if (size < 2) {
        printf("At least 2 processes are required.\n");
        MPI_Finalize();
        exit(1);
    }

    if (rank == 0) {
        int a[16];
        int i;
        for (i = 0; i < 16; i++) {
            a[i] = i;
        }

        /* Send a message to process 1 */
        MPI_Send(a, 16, MPI_INT, 1, 100, MPI_COMM_WORLD);
    }
    else if (rank == 1) {
        int b[16];
        MPI_Status stat;
        int i;
        
        /* Receive a message from process 0 */
        MPI_Recv(b, 16, MPI_INT, 0, 100, MPI_COMM_WORLD, &stat);

        for (i = 0; i < 16; i++) {
            printf("b[%d] = %d\n", i, b[i]);
        }
    }

    MPI_Finalize();
}
