module mod
    integer, allocatable, dimension(:) :: cherche
!$acc declare device_resident(cherche)

contains

    subroutine s_init()
        allocate(cherche(1:10))
    end subroutine s_init

    subroutine s_faire(i)
        !$acc routine seq

        integer, intent(IN) :: i

        cherche(i) = i
    end subroutine s_faire

    subroutine s_run()
        !$acc parallel loop vector default(present)
        do i = 1, 10
            call s_faire(i)
        end do
    end subroutine s_run
end module mod

program main
    use mod
    call s_init()
    call s_run()
end program main
