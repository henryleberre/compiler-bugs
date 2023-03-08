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