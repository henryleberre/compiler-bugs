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