#include <stdio.h>
#include <stdlib.h>

double* arrs;

#pragma acc declare create(arrs)

void m_mod_init() {
    arrs = (double*)malloc(10*sizeof(double));

    #pragma acc enter data create(arrs[0:9])

    #pragma acc update device(arrs[0:9])

    #pragma acc exit data delete(arrs)
}