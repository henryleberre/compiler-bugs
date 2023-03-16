module vars
    integer, allocatable, dimension(:) :: cherche
    !$acc declare create(cherche)

contains
    subroutine s_init()

         allocate(cherche(1:10))

   end subroutine s_init
end module vars

module mod
    use vars
contains
    subroutine s_faire(i)
        !$acc routine seq

        integer, intent(IN) :: i

        cherche(i) = i
    end subroutine s_faire
end module mod

program main
    use mod

    integer :: i

    call s_init()

    !$acc parallel loop present(cherche)
    do i = 1, 10
        call s_faire(i)
    end do

end program main
