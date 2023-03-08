# OLCFDEV-1416: Crusher: Cray Fortran + OpenACC bug

Dear Crusher support team,

 

I believe I have found a bug in the Cray Fortran compiler’s OpenACC implementation. I was able to create the following minimal reproducible example:

 
```f90
module mod

    integer, allocatable, dimension(:) :: cherche
    !$acc declare create(cherche)

contains

    subroutine s_run()
        allocate(cherche(1:10))
        !$acc enter data create(cherche(1:10))

        !$acc parallel loop default(present)
        do i = 1, 10
            cherche(i) = i
        end do

        !$acc update host(cherche(1:10))
        
        print*, cherche(:)

        !$acc exit data delete(cherche)

    end subroutine

end module mod

program main

    use mod

    call s_run()

end program main
```
 

The expected behavior of this program is to:

Dynamically allocate on the host and on the device an array named cherche with 10 elements.
On the device, set each value in the array to 1.
Copy the data from the device to the host.
Print the cherche array to verify it ran successfully.
Deallocate the cherche array.
 

Attempting to compile this code in debug mode leads to a compile-time error:

 
```console
henrylb@crusher:test $ module list

Currently Loaded Modules:

  1) craype-x86-trento                       8) craype/2.7.16          15) rocm/5.1.0
  2) libfabric/1.15.0.0                      9) cray-dsmml/0.2.2       16) craype-accel-amd-gfx90a
  3) craype-network-ofi                     10) cray-mpich/8.1.17      17) cmake/3.23.2
  4) perftools-base/22.06.0                 11) cray-libsci/22.06.1.3  18) cray-fftw/3.3.10.2
  5) xpmem/2.4.4-2.3_11.2__gff0e1d9.shasta  12) PrgEnv-cray/8.3.3      19) cray-hdf5/1.12.2.1
  6) cray-pmi/6.1.3                         13) xalt/1.3.0             20) cray-python/3.9.13.1
  7) cce/14.0.2                             14) DefApps/default        21) ninja/1.10.2



henrylb@crusher:test $ ftn -h acc -h msgs -h nomessage=1069 -e D test.f90 -o test

            cherche(i) = 1
ftn-7060 ftn: ERROR S_RUN, File = test.f90, Line = 11
  Unsupported OpenACC construct Fortran character -- t$14
ftn-7060 ftn: ERROR S_RUN, File = test.f90, Line = 11
  Unsupported OpenACC construct Fortran character -- t$13

Cray Fortran : Version 14.0.2 (20220720202147_ecfd9ef4dfd5696cd449133c0da0293d503c2f21)
Cray Fortran : Sun Jan 01, 2023  16:33:17
Cray Fortran : Compile time:  0.0142 seconds
Cray Fortran : 25 source lines
Cray Fortran : 2 errors, 0 warnings, 0 other messages, 0 ansi
Cray Fortran : "explain ftn-message number" gives more information about each message.
```
 

ftn-7060 is:

 
```console
henrylb@crusher:test $ explain ftn-7060

FATAL:  Unsupported OpenACC construct.

An unsupported OpenACC construct was encountered.
```
 

If I compile and run this code without -e D, the program compiles without any warnings, but a runtime error occurs:

 
```console
henrylb@crusher:test $ ftn -h acc -h msgs test.f90 -o test

        !$acc parallel loop default(present)
ftn-6405 ftn: ACCEL S_RUN, File = test.f90, Line = 9
  A region starting at line 9 and ending at line 12 was placed on the accelerator.

        do i = 1, 10
ftn-6430 ftn: ACCEL S_RUN, File = test.f90, Line = 10
  A loop starting at line 10 was partitioned across the threadblocks and the 256 threads within a threadblock.

Cray Fortran : Version 14.0.2 (20220720202147_ecfd9ef4dfd5696cd449133c0da0293d503c2f21)
Cray Fortran : Sun Jan 01, 2023  16:33:40
Cray Fortran : Compile time:  0.0392 seconds
Cray Fortran : 25 source lines
Cray Fortran : 0 errors, 0 warnings, 2 other messages, 0 ansi
henrylb@crusher:test $ ./test
:0:rocdevice.cpp            :2614: 281102691423 us: 54282: [tid:0x7ff1f9c47700] Device::callbackQueue aborting with error : HSA_STATUS_ERROR_MEMORY_FAULT: Agent attempted to access an inaccessible address. code: 0x2b

Aborted
```
 

I verified that this code runs properly on both NVIDIA and GNU 13 compilers:

 
```console
hberre3@wingtip-gpu3:~$ nvfortran -acc -Minfo=accel test.f90 -o test
s_run:
      7, Generating enter data create(cherche(1:10))
      9, Generating NVIDIA GPU code
         10, !$acc loop gang ! blockidx%x
     14, Generating update self(cherche(1:10))
     18, Generating exit data delete(cherche(:))
hberre3@wingtip-gpu3:~$ ./test
            1            1            1            1            1            1
            1            1            1            1

[hberre3@29:instinct]:new $ gfortran -o test -fopenacc '-foffload-options=-lgfortran\' -lm -foffload-options=amdgcn-amdhsa=-march=gfx90a -fno-exceptions test.f90 -lgfortran
[hberre3@30:instinct]:new $ ./test
           1           1           1           1           1           1           1           1           1           1
```

The full code I am working on getting running on Crusher is available here (with recent modifications): https://github.com/henryleberre/MFC. Thank you in advance for your help.

Sincerely,

Henry Le Berre

Computational Physics Group @ GT CSE