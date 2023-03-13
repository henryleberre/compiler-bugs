module transformations
    contains
    subroutine transform(u,v,w,result)
! Need to tell the compiler that this needs to be generated for the device (GPU)
   !$acc routine seq

! All global variables must be on the device. How to do this in OpenACC?
      use metric, only : xjac

      implicit none

      real*8 :: u, v, w, result

      result = u*xjac(1,1,1,1)+v*xjac(2,1,1,1) + &
               v*xjac(3,1,1,1)

   end subroutine transform
end module transformations

module metric

   implicit none

   real*8, pointer::xjac(:,:,:,:)
! Declaring an allocatable array as device_resident works
   !$acc declare device_resident(xjac)

end module metric
program main

   use transformations, only : transform
   use metric, only : xjac

   implicit none

   integer :: i, j, k, ijac
   integer :: imax = 1000, jmax = 1000, kmax = 1000
   real*8 :: u = 1.0d0, v = 1.0d0, w = 1.0d0
   real*8 :: result

! Filling in pointer array
   allocate(xjac(3,1000,1000,1000))
   !$acc parallel loop collapse(4) default(present)
   do k = 1, kmax
      do j = 1, jmax
         do i = 1, imax
            do ijac = 1,3
               xjac(ijac,i,j,k) = 1.0d0
            enddo
         enddo
      enddo
   enddo

! Starting a parallel region for a nested 3D loop
   !$acc parallel loop collapse(3)
   do k = 1, kmax
      do j = 1, jmax
         do i = 1, imax
! Calling a subroutine from within the parallel region
            call transform(u,v,w,result)
         enddo
      enddo
   enddo

end program
