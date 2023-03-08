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