import numpy as np
from scipy.interpolate import interp1d

class KF():
    '''
    Data for Kernel Table
    '''
    def __int__(self,FileName=None,gmax_eta=None):
        self.FileName = FileName
        self.gmax_eta = gmax_eta
        
        if self.FileName is None:
            print('FileName is not set in input.Kernel_Table.dat not EXIST?')
        
    def readkernl(self,FileName):
        '''
        Read Kernel Table file
        Input: Kernel Table file
        Returun: Kernel function-- omega(eta),eta_min,eta_max
        '''
        with open (FileName,'r') as kerfile:
            lines = kerfile.readlines()
            
        for i in range(0,len(lines)):
            line = lines[i]
            if 'END COMMENT' in line:
                ib = i+3
            if 'EOF' in line:
                ie = i
        line = " ".join([line.strip() for line in lines[ib:ie]])
        if 'EOF' in lines[ie]:
            print('Kernel Table '+FileName+' loaded')
        else:
            return Exception
        v1 = lines[ib-2].split()
        v2 = lines[ib-1].split()
        omega = np.array(line.split()).astype(np.float) 
        numpoints = np.int(v1[0])
        num_range = np.int(v1[1])
        gmax_eta = np.float(v1[2])
        d_eta= np.array(v1[3:]).astype(np.float)
        
        eta_min=d_eta[0]
        eta_max=gmax_eta
        
        
        d_points = np.array(v2[0:num_range]).astype(np.float)
        lmax_eta = np.array(v2[num_range:]).astype(np.float)
        
        # Evaluate the correspoding q-points
        eta=np.zeros(numpoints,dtype=float)
        for i in range(0,numpoints):
            for j in range(0,num_range):
                if (i <= d_points[0]):
                    eta[i]=d_eta[0]*(i+1.0)
                else:
                    if((i> sum(d_points[0:j])) and (i <= sum(d_points[0:j+1]))):
                        eta[i]=lmax_eta[j-1] + float(i-np.sum(d_points[0:j]))*d_eta[j]
                     # Interpolates the Omega(eta) function 
    
        order = 1
        omegaOFeta=interp1d(eta,omega,kind=order)
        
        return omegaOFeta, eta_min, eta_max    
