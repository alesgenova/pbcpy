# Drivers for LibXC

import numpy as np
from .field import DirectField
from .functionals import *
import pylibxc 
from pylibxc.functional import LibXCFunctional

def Get_LibXC_Input(density,do_sigma=True):
    if not isinstance(density,(DirectField)):
        raise TypeError("density must be a PBCpy DirectField")
    if density.rank != 1:
        raise AttributeError("Wrong rank")
    dim=np.shape(np.shape(density))[0]
    if dim > 4 or dim < 3:
        raise AttributeError("Wrong dimension of density input")
    rho=density[:,:,:].reshape(np.shape(density)[0]*np.shape(density)[1]*np.shape(density)[2])
    if do_sigma:
        sigma = density.sigma().reshape(np.shape(density)[0]*np.shape(density)[1]*np.shape(density)[2])
    inp = {}
    inp["rho"]=rho
    inp["sigma"]=sigma
    return inp


def Compute_LibXC(inp,func,spin):
    if not isinstance(inp,(dict)):
        raise AttributeError("LibXC Input must be a dictionary")
    if spin == 1:
        sspin="unpolarized"
    else:
        sspin="polarized"
    print("Computing "+func+" by LibXC")
    func=LibXCFunctional(func, sspin)
    return func.compute(inp)





def Get_LibXC_Output(out,density):
    if not isinstance(out,(dict)):
        raise TypeError("LibXC output must be a dictionary")
    if density.rank != 1:
        raise AttributeError("Wrong rank")
    dim=np.shape(np.shape(density))[0]
    if dim > 4 or dim < 3:
        raise AttributeError("Wrong dimension of density input")

    OutFunctional = Functional(name='LibXC')

    do_sigma = False
    if "vsigma" in out.keys():
        do_sigma = True

    if do_sigma:
        sigma = density.sigma().reshape(np.shape(density)[0]*np.shape(density)[1]*np.shape(density)[2])
    if "zk" in out.keys():
        edens = out["zk"].reshape(np.shape(density))

    if "vrho" in out.keys():
        vrho = DirectField(density.grid,rank=1,griddata_3d=out["vrho"].reshape(np.shape(density)))

    if "vsigma" in out.keys():
        vsigma = DirectField(density.grid,griddata_3d=out["vsigma"].reshape(np.shape(density)))

    if not do_sigma:
        OutFunctional.energydensity = DirectField(density.grid,rank=1,griddata_3d=edens*density)
        OutFunctional.potential     = DirectField(density.grid,rank=1,griddata_3d=vrho)
    else:
        grho = density.gradient(flag='supersmooth')
        rho_3 = grho.copy()
        rho_3[:,:,:,0] = density[:,:,:,0]
        rho_3[:,:,:,1] = density[:,:,:,0]
        rho_3[:,:,:,2] = density[:,:,:,0]
        prodotto=vsigma*grho 
        a = np.zeros(np.shape(prodotto))
        mask1 = np.where( grho > 1.0e-10 )
        mask2 = np.where( rho_3 > 1.0e-6 )
        a[mask1] = prodotto[mask1]
        a[mask2] = prodotto[mask2]
        vsigma_last = prodotto.divergence()
        v=vrho-2*vsigma_last
        OutFunctional.energydensity = DirectField(density.grid,rank=1,griddata_3d=np.real(edens*density))
        OutFunctional.potential     = DirectField(density.grid,rank=1,griddata_3d=np.real(v))

    return OutFunctional 


def XC(density,x_str,c_str,polarization):
    '''
     Output: 
        - Functional_XC: a PBCpy XC functional evaluated with LibXC
     Input:
        - density: a DirectField (rank=1)
        - x_str,c_str: strings like "gga_x_pbe" and "gga_c_pbe"
        - polarization: string like "polarized" or "unpolarized"
    '''
    if not isinstance(x_str, str):
        raise AttributeError("x_str and c_str must be LibXC functionals. Check pylibxc.util.xc_available_functional_names()")
    if not isinstance(c_str, str):
        raise AttributeError("x_str and c_str must be LibXC functionals. Check pylibxc.util.xc_available_functional_names()")
    if not isinstance(polarization, str):
        raise AttributeError("polarization must be a ``polarized`` or ``unpolarized``")
    if not isinstance(density,(DirectField)):
        raise AttributeError("density must be a rank-1 PBCpy DirectField")
    func_x = LibXCFunctional(x_str, polarization)
    func_c = LibXCFunctional(c_str, polarization)
    inp=Get_LibXC_Input(density)
    out_x = func_x.compute(inp)
    out_c = func_c.compute(inp)
    Functional_X = Get_LibXC_Output(out_x,density)
    Functional_C = Get_LibXC_Output(out_c,density)
    Functional_XC = Functional_X.sum(Functional_C)
    name = x_str[6:]+"_"+c_str[6:]
    Functional_XC.name = name.upper()
    return Functional_XC



def PBE(density,polarization):
    return XC(density=density,x_str='gga_x_pbe',c_str='gga_c_pbe',polarization='unpolarized')

def LDA(density,polarization):
    return XC(density=density,x_str='lda_x',c_str='lda_c_pz',polarization='unpolarized')



def KEDF(density,k_str,polarization):
    '''
     Output: 
        - Functional_KEDF: a PBCpy KEDF functional evaluated with LibXC
     Input:
        - density: a DirectField (rank=1)
        - k_str: strings like "gga_k_lc94"
        - polarization: string like "polarized" or "unpolarized"
    '''
    if not isinstance(k_str, str):
        raise AttributeError("k_str must be a LibXC functional. Check pylibxc.util.xc_available_functional_names()")
    if not isinstance(polarization, str):
        raise AttributeError("polarization must be a ``polarized`` or ``unpolarized``")
    if not isinstance(density,(DirectField)):
        raise AttributeError("density must be a rank-1 PBCpy DirectField")
    func_k = LibXCFunctional(k_str, polarization)
    inp=Get_LibXC_Input(density)
    out_k = func_k.compute(inp)
    Functional_KEDF = Get_LibXC_Output(out_k,density)
    name = k_str[6:]
    Functional_KEDF.name = name.upper()
    return Functional_KEDF


