# GNU [108895](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108895): [13.0.1 (exp)] Fortran + gfx90a !$acc update device produces a segfault.

The minimal reproducible sample (fortran + openacc).

gfortran configuration:

```console
[hberre3@96:instinct]:gcc-acc-test $ gfortran -v
Using built-in specs.
COLLECT_GCC=gfortran
COLLECT_LTO_WRAPPER=/nethome/hberre3/gcc-acc/libexec/gcc/x86_64-pc-linux-gnu/13.0.1/lto-wrapper
OFFLOAD_TARGET_NAMES=amdgcn-amdhsa
Target: x86_64-pc-linux-gnu
Configured with: /nethome/hberre3/temp-gcc-acc-work/build-gcc-amdgpu//gcc/configure --build=x86_64-pc-linux-gnu --host=x86_64-pc-linux-gnu --target=x86_64-pc-linux-gnu --enable-offload-targets=amdgcn-amdhsa=/nethome/hberre3/gcc-acc//amdgcn-amdhsa --enable-languages=c,c++,fortran,lto --disable-multilib --prefix=/nethome/hberre3/gcc-acc/
Thread model: posix
Supported LTO compression algorithms: zlib zstd
gcc version 13.0.1 20230219 (experimental) (GCC)
```

Compiled on:

```
GNU:  0263e9d5d84b4abbb53e73fbc8d72fd233764fc8 (master)
LLVM: llvmorg-13.0.1 (GitHub release)
```

Minimal reproducible (also found attached):

```f90
! Henry Le Berre <hberre3@gatech.edu> 

program p_main

    real(kind(0d0)), allocatable, dimension(:) :: arrs
    !$acc declare create(arrs)

    allocate(arrs(1000))
    !$acc enter data create(arrs(1000))
    !$acc update device(arrs(1:1000))

end program
```

Compiled with:

```console
gfortran -g -fopenacc -foffload-options=-march=gfx90a sample.f90 -o sample
```

Produces:

```console
[hberre3@102:instinct]:gcc-acc-test $ ./sample

Program received signal SIGSEGV: Segmentation fault - invalid memory reference.


 Backtrace for this error:
 #0  0x7fd01c643b1f in ???
 #1  0x7fd01c6c4ee9 in ???
 #2  0x7fd01bdf6007 in ???
 #3  0x7fd01bdd921f in ???
 #4  0x7fd01c1e5088 in hsa_memory_copy_wrapper
 	at /nethome/hberre3/temp-gcc-acc-work/build-gcc-amdgpu//gcc/libgomp/plugin/plugin-gcn.c:2958
 #5  0x7fd01c1eb1eb in GOMP_OFFLOAD_host2dev
 	at /nethome/hberre3/temp-gcc-acc-work/build-gcc-amdgpu//gcc/libgomp/plugin/plugin-gcn.c:3796
 #6  0x7fd01ce25cba in gomp_device_copy
 	at /nethome/hberre3/temp-gcc-acc-work/build-gcc-amdgpu//gcc/libgomp/target.c:234
 #7  0x7fd01ce25cba in gomp_copy_host2dev
	at /nethome/hberre3/temp-gcc-acc-work/build-gcc-amdgpu//gcc/libgomp/target.c:433
 #8  0x7fd01ce35596 in update_dev_host
 	at /nethome/hberre3/temp-gcc-acc-work/build-gcc-amdgpu//gcc/libgomp/oacc-mem.c:877
 #9  0x7fd01ce33142 in GOACC_update
 	at /nethome/hberre3/temp-gcc-acc-work/build-gcc-amdgpu//gcc/libgomp/oacc-parallel.c:678
 #10  0x400cad in p_main
 	at /nethome/hberre3/gcc-acc-test/sample.f90:10
 #11  0x400ced in main
 	at /nethome/hberre3/gcc-acc-test/sample.f90:3
 Segmentation fault (core dumped)
```

Observations:

1) If the length/size of the array were smaller (say 10 or 100) no segmentation fault is observed, possibly indicating silent R/W operations to memory we don't own.

2) On ORNL Summit's GCC 8.3.1 (nvptx), this sample does not produce a segfault. It was configured with:

```console
[henrylb@login4.summit ~]$ gcc -v
Using built-in specs.
COLLECT_GCC=gcc
COLLECT_LTO_WRAPPER=/usr/libexec/gcc/ppc64le-redhat-linux/8/lto-wrapper
OFFLOAD_TARGET_NAMES=nvptx-none
OFFLOAD_TARGET_DEFAULT=1
Target: ppc64le-redhat-linux
Configured with: ../configure --enable-bootstrap --enable-languages=c,c++,fortran,lto --prefix=/usr --mandir=/usr/share/man --infodir=/usr/share/info --with-bugurl=http://bugzilla.redhat.com/bugzilla --enable-shared --enable-threads=posix --enable-checking=release --enable-targets=powerpcle-linux --disable-multilib --with-system-zlib --enable-__cxa_atexit --disable-libunwind-exceptions --enable-gnu-unique-object --enable-linker-build-id --with-gcc-major-version-only --with-linker-hash-style=gnu --enable-plugin --enable-initfini-array --with-isl --disable-libmpx --enable-offload-targets=nvptx-none --without-cuda-driver --enable-gnu-indirect-function --enable-secureplt --with-long-double-128 --with-cpu-32=power8 --with-tune-32=power8 --with-cpu-64=power8 --with-tune-64=power8 --build=ppc64le-redhat-linux
Thread model: posix
gcc version 8.3.1 20191121 (Red Hat 8.3.1-5) (GCC)
```

3) If I translate this sample to C, no matter how large the array is, a segfault is not produced. Please excuse me if this C/OpenACC sample is invalid as I only use hip/Cuda when writing offloaded code in C/C++. This might indicate it is not an issue with libomp but I am not sure.

```c
#include <stdlib.h>

double* arrs;
#pragma acc declare create(arrs)

int main() {
    arrs = malloc(sizeof(double)*100000);
#pragma acc enter data create(arrs[1:100000])
#pragma acc update device(arrs[1:100000])
}
```

System: The gfx90a system I used for testing has AMD MI 210 GPUs and the nvptx ones have NVIDIA A100s/V100s.

Please let me know if there is anything more I can provide you with. I thank you in advance for your help!