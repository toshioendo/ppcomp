#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

int main(int argc, char *argv[])
{
  printf("Hello OpenMP World\n");

#pragma omp parallel
  {
    int i;
    for (i = 0; i < 5; i++) {
      printf("I'm %d-th thread out of %d threads\n",
	     omp_get_thread_num(), omp_get_num_threads());
      sleep(1);
    }
  }

  printf("Good Bye OpenMP World\n");

  return 0;
}
