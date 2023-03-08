# GNU [106643](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106643): [gfortran + OpenACC] Allocate in module causes refcount error

I built GCC 13 from the default branch with offloading support for the new AMD MI 210 GPUs by following the documented instructions. I ran into the following runtime error when running our offloaded code written in Fortran leveraging OpenACC:

```console
/nethome/hberre3/USERSCRATCH/build-gcc-amdgpu//gcc/libgomp/oacc-mem.c:1153: goacc_enter_data_internal: Assertion `n->refcount != REFCOUNT_INFINITY && n->refcount != REFCOUNT_LINK' failed.
```

I was able to create a minimal reproducible example for it:

p_main.f90
```f90
program p_main

    use m_macron

    ! ==========================================================================

    implicit none

    call s_vars_init()
    call s_vars_read()
    call s_macron_init()

end program p_main
```

m_macron.f90
```f90
module m_macron

    use m_vars

    implicit none

    real(kind(0d0)), allocatable, dimension(:) :: valls
    !$acc declare create(valls)

contains

    subroutine s_macron_init()

        integer :: i

        print*, "size=", size

        print*, "allocate(valls(1:size))"
        allocate(valls(1:size))

        print*, "acc enter data create(valls(1:size))"
        !$acc enter data create(valls(1:size))

        print*, "!$acc update device(valls(1:size))"
        valls(size) = size - 2
        !$acc update device(valls(1:size))

        print*, valls(1:size)

        print*, "acc exit data delete(valls)"
        !$acc exit data delete(valls)

    end subroutine s_macron_init

end module m_macron
```

m_vars.f90
```f90
module m_vars

    integer :: size

contains

    subroutine s_vars_init()

        size = -100

    end subroutine s_vars_init

    subroutine s_vars_read()

        ! Namelist of the global parameters which may be specified by user
        namelist /user_inputs/ size

        open (1, FILE=trim("in.inp"), FORM='formatted', ACTION='read', STATUS='old')
        read (1, NML=user_inputs); close (1)

    end subroutine s_vars_read

end module m_vars
```

in.inp
```
&user_inputs
size = 10
&end/
```

In order to generate this error, I had to create and dynamically allocate the array in another module. I initially wrote this in a single F90 file but the executable ran as expected. The error gets produced when running:
```f90
!$acc enter data create(valls(1:size))
```

Here is the full output when running the executable with the `GOMP_DEBUG` environment variable set:
```console
GOACC_data_start: mapnum=3, hostaddrs=0x7fff23d2efd0, size=0x7fff23d2efb0, kinds=0x603102
  GOACC_data_start: prepare mappings
  GOACC_data_start: mappings prepared
 size=          10
 allocate(valls(1:size))
 acc enter data create(valls(1:size))
main: /nethome/hberre3/USERSCRATCH/build-gcc-amdgpu//gcc/libgomp/oacc-mem.c:1153: goacc_enter_data_internal: Assertion `n->refcount != REFCOUNT_INFINITY && n->refcount != REFCOUNT_LINK' failed.

Program received signal SIGABRT: Process abort signal.

Backtrace for this error:
#0  0x7f4aa4fcdb1f in ???
#1  0x7f4aa4fcda9f in ???
#2  0x7f4aa4fa0e04 in ???
#3  0x7f4aa4fa0cd8 in ???
#4  0x7f4aa4fc63f5 in ???
#5  0x7f4aa57b79e5 in goacc_enter_data_internal
        at /nethome/hberre3/USERSCRATCH/build-gcc-amdgpu//gcc/libgomp/oacc-mem.c:1153
#6  0x7f4aa57b79e5 in goacc_enter_exit_data_internal
        at /nethome/hberre3/USERSCRATCH/build-gcc-amdgpu//gcc/libgomp/oacc-mem.c:1405
#7  0x7f4aa57b8aab in GOACC_enter_data
        at /nethome/hberre3/USERSCRATCH/build-gcc-amdgpu//gcc/libgomp/oacc-mem.c:1478
#8  0x4013b7 in __m_macron_MOD_s_macron_init
        at /nethome/hberre3/bug/m_macron.f90:22
#9  0x40182a in p_main
        at /nethome/hberre3/bug/p_main.f90:11
#10  0x401866 in main
        at /nethome/hberre3/bug/p_main.f90:3
run.sh: line 17: 3052573 Aborted                 (core dumped) ./main
```

I am not sure if this error is from libgomp or another part of the GCC codebase. I assume it is related to an issue with the scoping of the array, as it should be available throughout the entire program, per my reading of the OpenACC spec:

> The associated region is the implicit region associated with the function, subroutine, or program in which the directive appears. If the directive appears in the declaration section of a Fortran module subprogram or in a C or C++ global scope, the associated region is the implicit region for the whole program. 

Our main code currently doesn't call `!$acc enter data create` for dynamically allocated arrays since it relies on NVIDIA (/PGI) hooking into the `allocate` call on the CPU. I ran into the above error when converting our allocation/deallocation routines.

Here is the output of `gfortran -v`:
```console
[hberre3@8:instinct]:~ $ ~/tools/gcc/13/bin/gfortran -v
Using built-in specs.
COLLECT_GCC=/nethome/hberre3/tools/gcc/13/bin/gfortran
COLLECT_LTO_WRAPPER=/nethome/hberre3/tools/gcc/13/libexec/gcc/x86_64-pc-linux-gnu/13.0.0/lto-wrapper
OFFLOAD_TARGET_NAMES=amdgcn-amdhsa
Target: x86_64-pc-linux-gnu
Configured with: /nethome/hberre3/USERSCRATCH/build-gcc-amdgpu//gcc/configure --build=x86_64-pc-linux-gnu --host=x86_64-pc-linux-gnu --target=x86_64-pc-linux-gnu --enable-offload-targets=amdgcn-amdhsa=/nethome/hberre3/tools/gcc/13/amdgcn-amdhsa --enable-languages=c,c++,fortran,lto --disable-multilib --prefix=/nethome/hberre3/tools/gcc/13
Thread model: posix
Supported LTO compression algorithms: zlib zstd
gcc version 13.0.0 20220811 (experimental) (GCC)
```

The code is compiled with `/nethome/hberre3/tools/gcc/13/bin/gfortran -O0 -g -fopenacc '-foffload-options=-lgfortran -lm' -foffload-options=amdgcn-amdhsa=-march=gfx90a -fno-exceptions`.

This example runs with NVFortran on NVIDIA GPUs. Thank you for taking a look!

Here is a more minimal version:

m_mod.f90:
```f90
module m_mod

    real(kind(0d0)),allocatable, dimension(:) :: arrs
    !$acc declare create(arrs)

contains

    subroutine s_mod_init()

        allocate(arrs(10))
        
        !$acc enter data create(arrs(10))

        !$acc update device(arrs(10))

        !$acc exit data delete(arrs)

    end subroutine s_mod_init

end module m_mod
```


p_main.f90:
```f90
program p_main

    use m_mod
    
    call s_mod_init()

end program p_main
```

Here is a C version that runs without issue:

m_mod.c:
```c
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
```

p_main.c:
```c
extern void m_mod_init();

int main() {
    m_mod_init(); 
}
```
