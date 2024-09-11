#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>

int main(int argc, char *argv[])
{
#pragma omp parallel
#pragma omp single
    {
        printf("There are %d threads\n", omp_get_num_threads());
#pragma omp task
        {
            printf("I'm %d-th thread: Start A\n",
                   omp_get_thread_num());
            sleep(4);
            printf("I'm %d-th thread: End A\n",
                   omp_get_thread_num());
        }
    
#pragma omp task
        {
            printf("I'm %d-th thread: Start B\n",
                   omp_get_thread_num());
            sleep(2);
            printf("I'm %d-th thread: End B\n",
                   omp_get_thread_num());
        }
    
        printf("I'm %d-th thread: Start C\n",
               omp_get_thread_num());
        sleep(3);
        printf("I'm %d-th thread: End C\n",
               omp_get_thread_num());
    
#pragma omp taskwait
    
        printf("I'm %d-th thread: taskwait ended\n",
               omp_get_thread_num());
    } // parallel region ends

    return 0;
}
