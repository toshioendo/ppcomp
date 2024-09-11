#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

long time_diff_us(struct timeval st, struct timeval et)
{
    return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

long fib_r(int n)
{
    long f1, f2;
    if (n <= 1) return n;

#pragma omp task shared(f1)
    f1 = fib_r(n-1);
    
#pragma omp task shared(f2)
    f2 = fib_r(n-2);

#pragma omp taskwait

    return f1+f2;
}

long fib(int n)
{
    long ans;
#pragma omp parallel
#pragma omp single
    {
        ans = fib_r(n);
    }

    return ans;
}

int main(int argc, char *argv[])
{
    int n = 30;
    long ans;
    int i;

    if (argc >= 2) {
        n = atoi(argv[1]);
    }

    for (i = 0; i < 3; i++) {
        struct timeval st;
        struct timeval et;
        long us;
        double res;

        gettimeofday(&st, NULL); /* get start time */
        ans = fib(n);
        gettimeofday(&et, NULL); /* get start time */
        us = time_diff_us(st, et);

        printf("fib(%d) = %ld: fib took %ld us\n",
               n, ans, us);

    }

    return 0;
}
