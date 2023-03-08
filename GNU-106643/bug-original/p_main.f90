program p_main

    use m_macron

    ! ==========================================================================

    implicit none

    call s_vars_init()
    call s_vars_read()
    call s_macron_init()

end program p_main