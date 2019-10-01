!SUBROUTINE LDA_xc_F(rho,calcType,pot,ene)
   !!intent in
   !IMPLICIT NONE
   !REAL(DP)							:: rho(:,:,:)
   !character(*)                     :: calcType
   !REAL(DP)							:: pot(size(rho,1),size(rho,2),size(rho,3))
   !REAL(DP)							:: ene
   !!local
   !INTEGER(I4B)						:: kind
   !!-----------------------------------------------------------------------
   !!f2py intent(in) rho
   !!f2py intent(in) calcType
   !!f2py intent(out) pot
   !!f2py intent(out) ene
   !!-----------------------------------------------------------------------
   !if (calcType == 'Energy') then 
      !call LDA_xc_F_Energy(rho, ene)
   !else if (calcType == 'Potential') then 
      !call LDA_xc_F_Potential(rho, pot)
   !else 
      !call LDA_xc_F_Energy(rho, ene)
      !call LDA_xc_F_Potential(rho, pot)
   !endif
   !return
!END SUBROUTINE
SUBROUTINE LDA_xc_F_Energy(rho,ene)
   !intent in
   USE constants
   IMPLICIT NONE
   REAL(DP) :: PI=3.141592653589793238462643383279502884197d0
   REAL(DP)							:: rho(:,:,:)
   REAL(DP)							:: ene
   !local
   REAL(DP), DIMENSION(2) :: &
      a=(0.0311d0,  0.01555d0),&
      b=(-0.048d0,  -0.0269d0),&
      c=(0.0020d0,  0.0007d0),&
      d=(-0.0116d0, -0.0048d0),&
      gamma=(-0.1423d0, -0.0843d0),&
      beta1=(1.0529d0,  1.3981d0),&
      beta2=(0.3334d0,  0.2611d0)
   REAL(DP) :: OneThird = 1.0d0/3.0d0
   REAL(DP) :: FourThird = 4.0d0/3.0d0
   INTEGER(I4B)						:: i,j,k
   INTEGER(I4B)						:: n0,n1,n2
   REAL(DP)							:: Rxe,Rc0
   REAL(DP)							:: tmp,rs,rsSqrt
   !-----------------------------------------------------------------------
   !f2py intent(in) rho
   !f2py intent(out) ene
   !-----------------------------------------------------------------------
   n0 = size(rho,1)
   n1 = size(rho,2)
   n2 = size(rho,3)
   ene = 0.0d0
   !array1 = rho ** OneThird
   Rxe = -3.0/4.0 * (3.0/PI) ** OneThird
   Rc0 = (3.0d0/(4.0d0 * PI))** OneThird
   ene = Rxe * sum(rho ** FourThird)
   do k=1,n2
      do j=1,n1
         do i =1,n0
            tmp = rho(i, j, k) ** OneThird
            rs = Rc0 / tmp
            if (rs < 1) then
               !tmp = a(1) * log(rs) + b(1) + c(1) * rs * log(rs) + d(1) * rs
               tmp = (a(1)+ c(1) * rs) * log(rs) + b(1) + d(1) * rs
            else
               tmp = gamma(1) / (1.0+beta1(1) * SQRT(rs) + beta2(1) * rs)
            endif
            ene = ene + tmp * rho(i, j, k)
         enddo
      enddo
   enddo
   return
END SUBROUTINE

SUBROUTINE LDA_xc_F_Potential(rho,pot)
   !intent in
   USE constants
   IMPLICIT NONE
   REAL(DP) :: PI=3.141592653589793238462643383279502884197d0
   REAL(DP)							:: rho(:,:,:)
   REAL(DP)							:: pot(size(rho,1),size(rho,2),size(rho,3))
   !local
   REAL(DP), DIMENSION(2)           :: &
      a=(0.0311d0,  0.01555d0),&
      b=(-0.048d0,  -0.0269d0),&
      c=(0.0020d0,  0.0007d0),&
      d=(-0.0116d0, -0.0048d0),&
      gamma=(-0.1423d0, -0.0843d0),&
      beta1=(1.0529d0,  1.3981d0),&
      beta2=(0.3334d0,  0.2611d0)
   REAL(DP) :: OneThird = 1.0d0/3.0d0
   INTEGER(I4B)						:: i,j,k
   INTEGER(I4B)						:: n0,n1,n2
   REAL(DP)							:: Rx,Rc0,Rc1,Rc2,Rc3,Rc4,Rc5
   REAL(DP)							:: tmp,rs,rsSqrt
   !-----------------------------------------------------------------------
   !f2py intent(in) rho
   !f2py intent(out) pot
   !-----------------------------------------------------------------------
   n0 = size(rho,1)
   n1 = size(rho,2)
   n2 = size(rho,3)
   Rx =  -(3.0d0/PI)** OneThird
   Rc0 = (3.0d0/(4.0d0 * PI))** OneThird
   Rc1 = 2.0d0/3d0 * c(1)
   Rc2 = b(1)-OneThird * a(1)
   Rc3 = OneThird * (2d0 * d(1)-c(1))
   Rc4 = 7.0d0/6.0d0 * gamma(1) * beta1(1)
   Rc5 = 4.0d0/3.0d0 * gamma(1) * beta2(1)
   do k=1,n2
      do j=1,n1
         do i =1,n0
            tmp = rho(i, j, k) ** OneThird
            pot(i, j, k) = Rx * tmp
            rs = Rc0 / tmp
            if (rs < 1) then
               pot(i, j, k) = pot(i,j,k)+ LOG(rs) * (a(1)+ Rc1 * rs) + Rc2 + Rc3*rs
            else
               rsSqrt = SQRT(rs)
               pot(i, j, k) = pot(i, j, k) + (gamma(1)+ Rc4 * rsSqrt + Rc5 * rs)/ &
                  ( 1.0d0+beta1(1) * rsSqrt + beta2(1) * rs) ** 2
            endif
         enddo
      enddo
   enddo
   return
END SUBROUTINE
