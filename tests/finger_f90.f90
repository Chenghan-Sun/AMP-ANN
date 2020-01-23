module finger_functions
contains
  function Kronecker(ii, jj) result(kk)
    integer, intent(in) :: ii, jj
    integer :: kk

    if (ii.EQ.jj) then
      kk=1
    else
      kk=0
    endif
  end function Kronecker

  function dRij_dRml_vector(ii, jj, mm, ll) result(dRRv)
    integer, intent(in) :: ii, jj, mm, ll
    integer :: c1
    integer, dimension(3) :: dRRv

    dRRv(1)=0
    dRRv(2)=0
    dRRv(3)=0
    if ((mm.NE.ii).or.(mm.NE.jj)) then
      c1 = Kronecker(mm, jj) - Kronecker(mm, ii)
      dRRv(ll)=c1
    endif
  end function dRij_dRml_vector

  function dRij_dRml(ii, jj, wab, mm, dij) result(dRR)
    integer, intent(in) :: ii, jj, mm
    real, intent(in) :: dij
    real, intent(in) :: wab
    real :: dRR

    if ((ii.ne.jj).and.(mm.eq.ii)) then
      dRR = -1.0*(wab) / dij
    elseif ((ii.ne.jj).and.(mm.eq.jj)) then
      dRR = (wab) / dij
    else
      dRR = 0
    endif
  end function dRij_dRml

  function get_zz(elem) result(zz)
    character (len=3), intent(in) :: elem
    integer :: zz

    select case (elem)
    case("H")
      zz = 1
    case("He")
      zz = 2
    case("Li")
      zz = 3
    case("Be")
      zz = 4
    case("B")
      zz = 5
    case("C")
      zz = 6
    case("N")
      zz = 7
    case("O")
      zz = 8
    case("F")
      zz = 9
    case("Ne")
      zz = 10
    case("Na")
      zz = 11
    case("Mg")
      zz = 12
    case("Al")
      zz = 13
    case("Si")
      zz = 14
    case("P")
      zz = 15
    case("S")
      zz = 16
    case("Cl")
      zz = 17
    case("Ar")
      zz = 18
    case("K")
      zz = 19
    case("Ca")
      zz = 20
    case("Sc")
      zz = 21
    case("Ti")
      zz = 22
    case("V")
      zz = 23
    case("Cr")
      zz = 24
    case("Mn")
      zz = 25
    case("Fe")
      zz = 26
    case("Co")
      zz = 27
    case("Ni")
      zz = 28
    case("Cu")
      zz = 29
    case("Zn")
      zz = 30
    case("Ga")
      zz = 31
    case("Ge")
      zz = 32
    case("As")
      zz = 33
    case("Se")
      zz = 34
    case("Br")
      zz = 35
    case("Kr")
      zz = 36
    case("Rb")
      zz = 37
    case("Sr")
      zz = 38
    case("Y")
      zz = 39
    case("Zr")
      zz = 40
    case("Nb")
      zz = 41
    case("Mo")
      zz = 42
    case("Tc")
      zz = 43
    case("Ru")
      zz = 44
    case("Rh")
      zz = 45
    case("Pd")
      zz = 46
    case("Ag")
      zz = 47
    case("Cd")
      zz = 48
    case("In")
      zz = 49
    case("Sn")
      zz = 50
    case("Sb")
      zz = 51
    case("Te")
      zz = 52
    case("I")
      zz = 53
    case("Xe")
      zz = 54
    case("Cs")
      zz = 55
    case("Ba")
      zz = 56
    case("La")
      zz = 57
    case("Ce")
      zz = 58
    case("Pr")
      zz = 59
    case("Nd")
      zz = 60
    case("Pm")
      zz = 61
    case("Sm")
      zz = 62
    case("Eu")
      zz = 63
    case("Gd")
      zz = 64
    case("Tb")
      zz = 65
    case("Dy")
      zz = 66
    case("Ho")
      zz = 67
    case("Er")
      zz = 68
    case("Tm")
      zz = 69
    case("Yb")
      zz = 70
    case("Lu")
      zz = 71
    case("Hf")
      zz = 72
    case("Ta")
      zz = 73
    case("W")
      zz = 74
    case("Re")
      zz = 75
    case("Os")
      zz = 76
    case("Ir")
      zz = 77
    case("Pt")
      zz = 78
    case("Au")
      zz = 79
    case("Hg")
      zz = 80
    case("Tl")
      zz = 81
    case("Pb")
      zz = 82
    case("Bi")
      zz = 83
    case("Po")
      zz = 84
    case("At")
      zz = 85
    case("Rn")
      zz = 86
    case("Fr")
      zz = 87
    case("Ra")
      zz = 88
    case("Ac")
      zz = 89
    case("Th")
      zz = 90
    case("Pa")
      zz = 91
    case("U")
      zz = 92
    case("Np")
      zz = 93
    case("Pu")
      zz = 94
    case("Am")
      zz = 95
    case("Cm")
      zz = 96
    case("Bk")
      zz = 97
    case("Cf")
      zz = 98
    case("Es")
      zz = 99
    case("Fm")
      zz = 100
    case("Md")
      zz = 101
    case("No")
      zz = 102
    case("Lr")
      zz = 103
    case("Rf")
      zz = 104
    case("Db")
      zz = 105
    case("Sg")
      zz = 106
    case("Bh")
      zz = 107
    case("Hs")
      zz = 108
    case("Mt")
      zz = 109
    case("Ds")
      zz = 110
    end select

  end function get_zz
end module

program finger
  use finger_functions
  implicit NONE

  real :: vab(3), vac(3), vbc(3)
  real :: wab(3), wac(3), wbc(3)
  real :: dab, dac, dbc
  real :: c1(3), c2(3), c3(3)
  integer  :: nion, ii, jj, kk
  integer  :: aa, bb, cc, nn, oo, ss, tt
  integer  :: at, bt, ct, mm, ll, mmi
  integer :: mmt(3)
  integer  :: ntype
  integer, allocatable :: zz(:)
  integer, allocatable :: neighb(:,:,:)
  integer :: neighb_index, max_neighb
  integer :: type_rank(110)
  
  real, parameter :: pi = 3.1415926
  real :: cut_ab, cut_ac, cut_bc
  real :: cut_abp, cut_acp, cut_bcp
  real :: dRabdRml, dRacdRml, dRbcdRml
  integer :: dRabdRmlv(3), dRacdRmlv(3)
  real :: term, term1, term2, term3, term4, term5, term6, termc
  real :: termp, dterm, cuts, cos_theta_ijk
  real :: dCosthetadRml

  real, allocatable :: cov_radii(:), tau(:,:)
  character (len=32) :: Rc_arg, ntype_arg, nion_arg, calc_prime_arg
  character (len=3) :: elements
  integer :: calc_prime
  real :: Rc
  real :: eta_G2(4)
  real :: eta_G4, gamma_G4(2), zeta_G4(2)
  real, allocatable :: ridge_G2(:,:,:), ridge_G4(:,:,:,:,:) 
  real, allocatable :: ridge_G2p(:,:,:,:,:), ridge_G4p(:,:,:,:,:,:,:)

  integer, allocatable :: type_index(:)
  integer, allocatable :: num_neighb(:)

  eta_G2(1)=0.05
  eta_G2(2)=4.0
  eta_G2(3)=20.0
  eta_G2(4)=80.0
  eta_G4=0.005
  gamma_G4(1)=1.0
  gamma_G4(2)=-1.0
  zeta_G4(1)=1.0
  zeta_G4(2)=4.0

  Call getarg(1, nion_arg)
  Call getarg(2, ntype_arg)
  Call getarg(3, Rc_arg)
  Call getarg(4, calc_prime_arg)

  read( nion_arg, * )  nion
  read( ntype_arg, * ) ntype
  read( Rc_arg, * ) Rc
  read( calc_prime_arg, * ) calc_prime

  Allocate(cov_radii(nion),tau(3,nion),zz(nion),num_neighb(nion),type_index(nion))
  Allocate(ridge_G2(nion,ntype,4), ridge_G4(nion,ntype,ntype,2,2))
  Allocate(ridge_G2p(nion,nion,3,ntype,4), ridge_G4p(nion,nion,3,ntype,ntype,2,2))

  do aa = 1, nion
    cov_radii(aa)=0
    tau(1,aa)=0
    tau(2,aa)=0
    tau(3,aa)=0
    zz(aa)=0
    num_neighb(aa)=0
    type_index(aa)=0
  enddo

  do aa = 1, nion
    do bb = 1, ntype
      do ss = 1, 4
        ridge_G2(aa,bb,ss)=0
        do mm = 1, nion
          do ll = 1, 3
            ridge_G2p(aa,mm,ll,bb,ss)=0
          enddo !ll
        enddo !bb
      enddo !ss
      do cc = 1, ntype
        do ss = 1, 2
          do tt = 1, 2
            ridge_G4(aa,bb,cc,ss,tt)=0
            do mm = 1, nion
              do ll = 1, 3
                ridge_G4p(aa,mm,ll,bb,cc,ss,tt)=0
              enddo !dd
            enddo !bb
          enddo !tt
        enddo !ss
      enddo !kk
    enddo !jj
  enddo !ii

  open(unit=10, file='elements')

  do aa = 1, nion
    read(10,*) elements
    zz(aa)=get_zz(elements)
  end do
  close(unit=10)

  ii=0
  Open (unit=11, file='element_alist')
  do aa = 1, ntype
    read(11,*) elements
    bb=get_zz(elements)
    type_rank(bb)=aa
  enddo
  close(unit=11)

  do aa=1,nion
    type_index(aa)=type_rank(zz(aa))
  enddo

  Open (unit=12, file='cell_matrix')
  Read(12,*) c1
  Read(12,*) c2
  Read(12,*) c3
  close(unit=12)

  Open (unit=13, file='positions')
  do aa=1,nion
    Read(13,*) tau(1,aa),tau(2,aa),tau(3,aa)
    num_neighb(aa)=0
  enddo
  close(unit=13)

  Rc=6.5
  max_neighb=0
  do aa = 1,nion
    num_neighb(aa)=0
    do bb = 1,nion
      vab(:) = (tau(1,bb)-tau(1,aa))*c1(:) + (tau(2,bb)-tau(2,aa))*c2(:) + (tau(3,bb)-tau(3,aa))*c3(:)
      do ii = -1, 1
        do jj = -1, 1
          do kk = -1, 1
            if ((aa.ne.bb).or.((abs(ii)+abs(jj)+abs(kk)).ne.0)) then
              wab(:) = vab(:) + ii*c1 + jj*c2 + kk*c3
              dab = sqrt(DOT_PRODUCT(wab,wab))
              if( dab.LT.Rc ) then
                num_neighb(aa)=num_neighb(aa) + 1
              endif     
            endif
          enddo    ! kk  loop
        enddo    ! jj loop
      enddo    ! ii loop
    enddo    ! bb loop
    if( num_neighb(aa).GT.max_neighb ) then
      max_neighb=num_neighb(aa)
    endif
  enddo   ! aa loop

  Allocate(neighb(nion,max_neighb,4))
  do aa = 1,nion
    do nn = 1,max_neighb
      do ss = 1,4
        neighb(aa,nn,ss)=0
      enddo
    enddo
  enddo

  ! create neighborlists
  do aa = 1,nion
    neighb_index=1
    do bb = 1,nion
      vab(:) = (tau(1,bb)-tau(1,aa))*c1(:) + (tau(2,bb)-tau(2,aa))*c2(:) + (tau(3,bb)-tau(3,aa))*c3(:)
      do ii = -1, 1
        do jj = -1, 1
          do kk = -1, 1
            if ((aa.ne.bb).or.((abs(ii)+abs(jj)+abs(kk)).ne.0)) then
              wab(:) = vab(:) + ii*c1 + jj*c2 + kk*c3
              dab = sqrt(DOT_PRODUCT(wab,wab))
              if( dab.LT.Rc ) then
                neighb(aa,neighb_index,1)=bb
                neighb(aa,neighb_index,2)=ii
                neighb(aa,neighb_index,3)=jj
                neighb(aa,neighb_index,4)=kk
                neighb_index=neighb_index + 1
              endif
            endif
          enddo    ! kk  loop
        enddo    ! jj loop
      enddo    ! ii loop
    enddo    ! bb loop
  enddo   ! aa loop

  open(21, file = 'neighbors', status = 'unknown')
  do aa = 1,nion
    do nn = 1,num_neighb(aa)
      Write(21,*) aa, neighb(aa,nn,1), neighb(aa,nn,2), neighb(aa,nn,3), neighb(aa,nn,4)
    enddo
  enddo
  close(21)

  ! create fingerprints
  do aa = 1,nion
    at=type_index(aa)
    do nn = 1,num_neighb(aa)
      bb=neighb(aa,nn,1)
      bt=type_index(bb)
      ii=neighb(aa,nn,2)
      jj=neighb(aa,nn,3)
      kk=neighb(aa,nn,4)
      vab(:) = (tau(1,bb)-tau(1,aa))*c1(:) + (tau(2,bb)-tau(2,aa))*c2(:) + (tau(3,bb)-tau(3,aa))*c3(:)
      wab(:) = vab(:) + ii*c1 + jj*c2 + kk*c3
      dab = sqrt(DOT_PRODUCT(wab,wab))
      dterm=dab**2 / Rc**2
      cut_ab=0.5 * (cos(pi * dab / Rc) + 1.)
      if (calc_prime.eq.1) then
        cut_abp=-0.5 * pi / Rc * sin(pi * dab / Rc)
      endif
      do ss = 1, 4
        term=exp(-1 * eta_G2(ss) * dterm ) * cut_ab
        ridge_G2(aa,bt,ss) = ridge_G2(aa,bt,ss) + term

        if (calc_prime.eq.1) then      
          termp=(-2) * eta_G2(ss) * dab * cut_ab / Rc ** 2 + cut_abp        
          mmt(1:2) = (/ aa, bb /)
          do mmi = 1,2   ! mm must be aa or bb for dRabdRml to be nonzero, will just run those
            mm=mmt(mmi)
            do ll = 1,3
              dRabdRml=dRij_dRml(aa,bb,wab(ll),mm,dab)
              ridge_G2p(aa,mm,ll,bt,ss)=ridge_G2p(aa,mm,ll,bt,ss) + &
                & term / cut_ab * termp * dRabdRml
            enddo ! ll
          enddo ! mmi
        endif
      enddo ! ss


      do oo = nn+1,num_neighb(aa)
        cc=neighb(aa,oo,1)
        ct=type_index(cc)
        ii=neighb(aa,oo,2)
        jj=neighb(aa,oo,3)
        kk=neighb(aa,oo,4)
        vac(:) = (tau(1,cc)-tau(1,aa))*c1(:) + (tau(2,cc)-tau(2,aa))*c2(:) + (tau(3,cc)-tau(3,aa))*c3(:)
        wac(:) = vac(:) + ii*c1 + jj*c2 + kk*c3
        vbc(:) = (tau(1,cc)-tau(1,bb))*c1(:) + (tau(2,cc)-tau(2,bb))*c2(:) + (tau(3,cc)-tau(3,bb))*c3(:)
        wbc(:) = vbc(:) + ii*c1 + jj*c2 + kk*c3
        dac = sqrt(DOT_PRODUCT(wac,wac))
        dbc = sqrt(DOT_PRODUCT(wbc,wbc))
        cos_theta_ijk = DOT_PRODUCT(wab, wac) / dab / dac
        cut_bc=0.5 * (cos(pi * dbc / Rc) + 1)
        cut_ac=0.5 * (cos(pi * dac / Rc) + 1)
        dterm=(dab**2 + dac**2 + dbc**2) / Rc**2
        cuts=cut_ab * cut_ac * cut_bc

        if (calc_prime.eq.1) then
          cut_acp=-0.5 * pi / Rc * sin(pi * dac / Rc)
          cut_bcp=-0.5 * pi / Rc * sin(pi * dbc / Rc)
        endif

        do ss = 1,2
          do tt = 1,2
            termc = 1.0 + gamma_G4(tt)*cos_theta_ijk
            term1 = exp(-1.0 * eta_G4 * dterm)
            term = ( termc )**zeta_G4(ss) * term1 * cuts * 2**(1-zeta_G4(ss))
            ridge_G4(aa,bt,ct,ss,tt) = ridge_G4(aa,bt,ct,ss,tt) + term

            if (calc_prime.eq.1) then
              if ( abs (zeta_G4(ss) - 1.0) > 1.0D-4 ) then
                term1 = term1 * termc**(zeta_G4(ss)-1)
              endif

              ! if mm is not aa, bb, or cc, then these will all be 0.
              mmt(1:3) = (/ aa, bb, cc /)
              do mmi = 1, 3
                do ll = 1, 3
                  mm=mmt(mmi)

                  dCosthetadRml=0
                  dRabdRmlv=dRij_dRml_vector(aa,bb,mm,ll)
                  if ((abs(dRabdRmlv(1))+abs(dRabdRmlv(2))+abs(dRabdRmlv(3))).GT.0) then
                    dCosthetadRml=dCosthetadRml+DOT_PRODUCT(dRabdRmlv(:), wac(:)) / (dab * dac)
                  endif
                  dRacdRmlv=dRij_dRml_vector(aa,cc,mm,ll)
                  if ((abs(dRacdRmlv(1))+abs(dRacdRmlv(2))+abs(dRacdRmlv(3))).GT.0) then
                    dCosthetadRml=dCosthetadRml+DOT_PRODUCT(wab(:), dRacdRmlv(:)) / (dab * dac)
                  endif
                  dRabdRml=dRij_dRml(aa, bb, wab(ll), mm, dab)
                  if (abs(dRabdRml) > 1D-4) then
                    dCosthetadRml=dCosthetadRml-DOT_PRODUCT(wab(:), wac(:))*dRabdRml / (dab**2 * dac)
                  endif
                  dRacdRml=dRij_dRml(aa, cc, wac(ll), mm, dac)
                  if (abs(dRacdRml) > 1D-4) then
                    dCosthetadRml=dCosthetadRml-DOT_PRODUCT(wab(:), wac(:))*dRacdRml / (dab * dac**2)
                  endif

                  dRbcdRml=dRij_dRml(bb, cc, wbc(ll), mm, dbc)

                  term2 = 0
                  term2 = term2 + gamma_G4(tt) * zeta_G4(ss) * dCosthetadRml
                  term2 = term2 + (-2)*termc*eta_G4*dab*dRabdRml/Rc**2
                  term2 = term2 + (-2)*termc*eta_G4*dac*dRacdRml/Rc**2
                  term2 = term2 + (-2)*termc*eta_G4*dbc*dRbcdRml/Rc**2 

                  term3 = cuts * term2
                  term4 = cut_abp * dRabdRml * cut_ac * cut_bc
                  term5 = cut_ab * cut_acp * dRacdRml * cut_bc
                  term6 = cut_ab * cut_ac * cut_bcp * dRbcdRml

                  ridge_G4p(aa,mm,ll,bt,ct,ss,tt)=ridge_G4p(aa,mm,ll,bt,ct,ss,tt) + &
                      & (term1 * (term3 + termc * (term4 + term5 + term6)) )*2**(1.0-zeta_G4(ss))

                enddo    ! ll loop
              enddo    ! mmi loop
            endif
          enddo    ! tt loop 
        enddo    ! ss loop
      enddo    ! oo loop
    enddo    ! nn loop
  enddo   ! aa loop

  open(22, file = 'fingerprints', status = 'unknown')
  do aa = 1, nion
    do ss = 1, 4    
      do bb = 1, ntype
        ! Write(22,*) aa, bb, 0, ss, 0, ridge_G2(aa,bb,ss)
        Write(22,*) ridge_G2(aa,bb,ss)
      enddo
    enddo
    do ss = 1, 2
      do tt = 1, 2
        do bb = 1, ntype
          do cc = bb, ntype
            if (bb.NE.cc) then
              ! Write(22,*) aa, bb, cc, ss, tt, ridge_G4(aa,bb,cc,ss,tt)+ridge_G4(aa,cc,bb,ss,tt)
              Write(22,*) ridge_G4(aa,bb,cc,ss,tt)+ridge_G4(aa,cc,bb,ss,tt)
            else
              ! Write(22,*) aa, bb, cc, ss, tt, ridge_G4(aa,bb,cc,ss,tt)
              Write(22,*) ridge_G4(aa,bb,cc,ss,tt)
            endif
          enddo
        enddo
      enddo
    enddo
  enddo
  close(22)

  if (calc_prime.eq.1) then
    open(23, file = 'primes', status = 'unknown')
    do aa = 1,nion
      do nn = 1,num_neighb(aa)+1
        if (nn.eq.1) then
          mm=aa
        else
          mm=neighb(aa,nn-1,1)
        endif
        do ll = 1, 3
          do ss = 1, 4
            do bb = 1, ntype
              Write(23,*) aa, mm, ll, ridge_G2p(aa,mm,ll,bb,ss)
              ! Write(23,*) ridge_G2p(aa,mm,ll,bb,ss)
            enddo ! bb
          enddo ! ss
          do ss = 1, 2
            do tt = 1, 2
              do bb = 1, ntype
                do cc = bb, ntype
                  if (bb.NE.cc) then
                    Write(23,*) aa, mm, ll, ridge_G4p(aa,mm,ll,bb,cc,ss,tt)+ridge_G4p(aa,mm,ll,cc,bb,ss,tt)
                    ! Write(23,*) ridge_G4p(aa,mm,ll,bb,cc,ss,tt)+ridge_G4p(aa,mm,ll,cc,bb,ss,tt)
                  else
                    Write(23,*) aa, mm, ll, ridge_G4p(aa,mm,ll,bb,cc,ss,tt)
                    ! Write(23,*) ridge_G4p(aa,mm,ll,bb,cc,ss,tt)
                  endif
                enddo ! cc
              enddo ! bb
            enddo ! tt
          enddo ! ss
        enddo ! ll
      enddo ! nn (mm)
    enddo ! aa
    close(23)
  endif

stop
end
