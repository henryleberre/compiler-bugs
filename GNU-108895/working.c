#include <stdlib.h>

double* arrs;
#pragma acc declare create(arrs)

int main() {
    arrs = malloc(sizeof(double)*100000);
#pragma acc enter data create(arrs[1:100000])
#pragma acc update device(arrs[1:100000])
}