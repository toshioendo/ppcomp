#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank;
    int size;
    char hostname[64];
    int i;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(hostname, 63);

    for (i = 0; i < 5; i++) {
        printf("Hello MPI World. I'm %d-th process out of %d processes. I'm living in %s\n", 
               rank, size, hostname);
        sleep(1);
    }

    MPI_Finalize();
    return 0;
}
