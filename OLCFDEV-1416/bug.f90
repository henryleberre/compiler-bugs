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