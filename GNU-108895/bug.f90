! Henry Le Berre <hberre3@gatech.edu> 

program p_main

    real(kind(0d0)), allocatable, dimension(:) :: arrs
    !$acc declare create(arrs)

    allocate(arrs(1000))
    !$acc enter data create(arrs(1000))
    !$acc update device(arrs(1:1000))

end program
