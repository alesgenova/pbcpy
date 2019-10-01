MODULE math_f
CONTAINS
   SUBROUTINE multiply(arr1,arr2,results)
      USE constants
      REAL(DP)							:: arr1(:,:,:,:)
      REAL(DP)							:: arr2(:,:,:,:)
      REAL(DP)							:: results(size(arr1,1),size(arr1,2),size(arr1,3),size(arr1,4))
      !local
      INTEGER(I4B)						:: i,j,k,l
      INTEGER(I4B)						:: n1,n2,n3,n4
      !-----------------------------------------------------------------------
      !f2py intent(in)  arr1,arr2
      !f2py intent(out) results
      !-----------------------------------------------------------------------
      n1 = size(arr1,1)
      n2 = size(arr1,2)
      n3 = size(arr1,3)
      n4 = size(arr1,4)
      !$OMP PARALLEL DO PRIVATE(l,k,j,i)
      do l = 1, n4
         do k = 1, n3
            do j = 1, n2
               do i = 1, n1
                  results(i,j,k,l) = arr1(i,j,k,l)* arr2(i,j,k,l)
               enddo
            enddo
         enddo
      enddo
      !$OMP END PARALLEL DO
   END SUBROUTINE multiply

   SUBROUTINE add(arr1,arr2,results)
      USE constants
      REAL(DP)							:: arr1(:,:,:,:)
      REAL(DP)							:: arr2(:,:,:,:)
      REAL(DP)							:: results(size(arr1,1),size(arr1,2),size(arr1,3),size(arr1,4))
      !local
      INTEGER(I4B)						:: i,j,k,l
      INTEGER(I4B)						:: n1,n2,n3,n4
      !-----------------------------------------------------------------------
      !f2py intent(in)  arr1,arr2
      !f2py intent(out) results
      !-----------------------------------------------------------------------
      n1 = size(arr1,1)
      n2 = size(arr1,2)
      n3 = size(arr1,3)
      n4 = size(arr1,4)
      !$OMP PARALLEL DO PRIVATE(l,k,j,i)
      do l = 1, n4
         do k = 1, n3
            do j = 1, n2
               do i = 1, n1
                  results(i,j,k,l) = arr1(i,j,k,l) + arr2(i,j,k,l)
               enddo
            enddo
         enddo
      enddo
      !$OMP END PARALLEL DO
   END SUBROUTINE add

   SUBROUTINE power(array,pow,results)
      USE constants
      REAL(DP)							:: array(:,:,:,:)
      REAL(DP)							:: results(size(array,1),size(array,2),size(array,3),size(array,4))
      REAL(DP)							:: pow
      !-----------------------------------------------------------------------
      !f2py intent(in)  array
      !f2py intent(in)  pow
      !f2py intent(out) results
      !-----------------------------------------------------------------------
      call power_4(array,pow,results)
   END SUBROUTINE power

   SUBROUTINE power_3(array,pow,results)
      USE constants
      !intent inout
      REAL(DP)							:: array(:,:,:)
      REAL(DP)							:: results(size(array,1),size(array,2),size(array,3))
      REAL(DP)							:: pow
      !local
      INTEGER(I4B)						:: i,j,k
      INTEGER(I4B)						:: n1,n2,n3
      !-----------------------------------------------------------------------
      !f2py intent(in)  array
      !f2py intent(in)  pow
      !f2py intent(out) results
      !-----------------------------------------------------------------------
      n1 = size(array,1)
      n2 = size(array,2)
      n3 = size(array,3)
      !$OMP PARALLEL DO PRIVATE(k,j,i)
      do k = 1, n3
         do j = 1, n2
            do i = 1, n1
               results(i,j,k) = array(i,j,k)** pow
            enddo
         enddo
      enddo
      !$OMP END PARALLEL DO
   END SUBROUTINE power_3

   SUBROUTINE power_4(array,pow,results)
      USE constants
      !intent inout
      REAL(DP)							:: array(:,:,:,:)
      REAL(DP)							:: results(size(array,1),size(array,2),size(array,3),size(array,4))
      REAL(DP)							:: pow
      !local
      INTEGER(I4B)						:: i,j,k,l
      INTEGER(I4B)						:: n1,n2,n3,n4
      !-----------------------------------------------------------------------
      !f2py intent(in)  array
      !f2py intent(in)  pow
      !f2py intent(out) results
      !-----------------------------------------------------------------------
      n1 = size(array,1)
      n2 = size(array,2)
      n3 = size(array,3)
      n4 = size(array,4)
      !$OMP PARALLEL DO PRIVATE(l,k,j,i)
      do l = 1, n4
         do k = 1, n3
            do j = 1, n2
               do i = 1, n1
                  results(i,j,k,l) = array(i,j,k,l)** pow
               enddo
            enddo
         enddo
      enddo
      !$OMP END PARALLEL DO
   END SUBROUTINE power_4

   SUBROUTINE cbrt_1(array,results)
      USE constants
      !intent inout
      REAL(DP)							:: array(:)
      REAL(DP)							:: results(size(array,1))
      !local
      INTEGER(I4B)						:: i
      INTEGER(I4B)						:: n1
      REAL(DP)							:: OneThird=1.d0/3.d0
      !-----------------------------------------------------------------------
      !f2py intent(in)  array
      !f2py intent(out) results
      !-----------------------------------------------------------------------
      n1 = size(array,1)
      !$OMP PARALLEL DO PRIVATE(i)
      do i = 1, n1
         results(i) = array(i)**OneThird
      enddo
      !$OMP END PARALLEL DO
   END SUBROUTINE cbrt_1

   SUBROUTINE cbrt_2(array,results)
      USE constants
      !intent inout
      REAL(DP)							:: array(:,:)
      REAL(DP)							:: results(size(array,1),size(array,2))
      !local
      INTEGER(I4B)						:: i,j
      INTEGER(I4B)						:: n1,n2
      REAL(DP)							:: OneThird=1.d0/3.d0
      !-----------------------------------------------------------------------
      !f2py intent(in)  array
      !f2py intent(out) results
      !-----------------------------------------------------------------------
      n1 = size(array,1)
      n2 = size(array,2)
      !$OMP PARALLEL DO PRIVATE(j,i)
      do j = 1, n2
         do i = 1, n1
            results(i,j) = array(i,j)**OneThird
         enddo
      enddo
      !$OMP END PARALLEL DO
   END SUBROUTINE cbrt_2

   SUBROUTINE cbrt_3(array,results)
      USE constants
      !intent inout
      REAL(DP)							:: array(:,:,:)
      REAL(DP)							:: results(size(array,1),size(array,2),size(array,3))
      !local
      INTEGER(I4B)						:: i,j,k
      INTEGER(I4B)						:: n1,n2,n3
      REAL(DP)							:: OneThird=1.d0/3.d0
      !-----------------------------------------------------------------------
      !f2py intent(in)  array
      !f2py intent(out) results
      !-----------------------------------------------------------------------
      n1 = size(array,1)
      n2 = size(array,2)
      n3 = size(array,3)
      !$OMP PARALLEL DO PRIVATE(k,j,i)
      do k = 1, n3
         do j = 1, n2
            do i = 1, n1
               results(i,j,k) = array(i,j,k)**OneThird
            enddo
         enddo
      enddo
      !$OMP END PARALLEL DO
   END SUBROUTINE cbrt_3

   SUBROUTINE cbrt_4(array,results)
      USE constants
      !intent inout
      REAL(DP)							:: array(:,:,:,:)
      REAL(DP)							:: results(size(array,1),size(array,2),size(array,3),size(array,4))
      !local
      INTEGER(I4B)						:: i,j,k,l
      INTEGER(I4B)						:: n1,n2,n3,n4
      REAL(DP)							:: OneThird=1.d0/3.d0
      !-----------------------------------------------------------------------
      !f2py intent(in)  array
      !f2py intent(out) results
      !-----------------------------------------------------------------------
      n1 = size(array,1)
      n2 = size(array,2)
      n3 = size(array,3)
      n4 = size(array,4)
      !$OMP PARALLEL DO PRIVATE(l,k,j,i)
      do l = 1, n4
         do k = 1, n3
            do j = 1, n2
               do i = 1, n1
                  results(i,j,k,l) = array(i,j,k,l)**OneThird
               enddo
            enddo
         enddo
      enddo
      !$OMP END PARALLEL DO
   END SUBROUTINE cbrt_4
END MODULE
