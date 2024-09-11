#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

long time_diff_us(struct timeval st, struct timeval et)
{
    return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

double pi(int n)
{
    int i;
    double sum = 0.0;
    double dx = 1.0 / (double)n;

#pragma omp parallel for
    for (i = 0; i < n; i++) {
        double x;
        double y;
        x = (double)i * dx;
        y = sqrt(1.0 - x*x);
        
        sum += dx*y;
    }

    return 4.0*sum;
}

int main(int argc, char *argv[])
{
    int n;
    int i;

    if (argc < 2) {
        printf("Specify #divisions.\n");
        exit(1);
    }

    n = atoi(argv[1]);

    /* Repeat same computation for 3 times */
    for (i = 0; i < 3; i++) {
        struct timeval st;
        struct timeval et;
        long us;
        double res;

        gettimeofday(&st, NULL); /* get start time */
        res = pi(n);
        gettimeofday(&et, NULL); /* get end time */

        us = time_diff_us(st, et);
        printf("Result=%.15lf: Pi took %ld us --> %lf Msamples/sec\n",
               res, us, (double)n/(double)us);
    }
    return 0;
}
