.SUFFIXES: .f90 .o

FC = ifort
PYFC = f2py --fcompiler=intelem # or gnu95
PYCC = f2py --compiler=intelem # or unix
OPTS = -O2 #OPTS += -fopenmp  or -qopenmp
#OPTS += -r8 or -fopenmp -fdefault-real-8 -fdefault-double-8
LOPTS = 
# if using OpenMP:
	#LOPTS = -liomp5 -lpthread or
	#LOPTS = -lgomp
export CFLAGS = -O2
BASE = 
FUNC = math_f lda_xc
ALL = $(BASE) $(FUNC)
Build=build

default: pbcpyF2PY

pbcpyF2PY: $(ALL)

$(BASE):%.o : %.f90
	$(FC) $(OPTS) $(LOPTS) -c $< -o $@ -module $(Build)

$(FUNC):% : %.f90
	$(PYFC) --opt="$(OPTS)" $(LOPTS) -m $@ -c constants.f90 $< --build-dir $(Build)

clean :
	rm -rf *.so *.o *.mod $(Build)
