#==============================================================================
# -*- encoding: utf-8 -*-
#==============================================================================

#==============================================================================
# Módulos Importados do Python / Devito / Examples
#==============================================================================

#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy as np
import math  as mt
import sys
#==============================================================================

#==============================================================================
class elasticodevitodamp:
#==============================================================================

#==============================================================================    
    def __init__(self,test):
                
        self.test = test
        self.VP, self.VS, self.RHO = self.geramt1(test)
        self.BRHO, self.LAM, self.MU, self.RO = self.geramt2(test)
        self.D0 = self.geramdamp(test,self.VP)
        self.Mpesosx, self.Mpesosy, self.Mpesosz = gerapesos(self,test)
#==============================================================================

#==============================================================================        
    def geramt1(self,test):
                
        npmlx = test.npmlx
        npmly = test.npmly
        npmlz = test.npmlz       
    
        vel_vp, vel_vs, rho = test.get_model()  

        npmlz_top = npmlz        
            
        VP = np.pad(vel_vp,
                   ((npmlx,npmlx), 
                    (npmly,npmly),
                    (npmlz_top,npmlz)),
                    'edge')
        VS = np.pad(vel_vs,
                    ((npmlx,npmlx), 
                     (npmly,npmly),
                     (npmlz_top,npmlz)),
                     'edge')
        RHO = np.pad(rho,
                    ((npmlx,npmlx), 
                     (npmly,npmly),
                     (npmlz_top,npmlz)),
                     'edge') 
      
        return VP, VS, RHO
#==============================================================================

#==============================================================================        
    def geramt2(self,test):
    
        VP   = self.VP
        VS   = self.VS
        RHO  = self.RHO   
        
        BRHO = 1/RHO
        LAM  = (VP**2-2.0*VS**2)/BRHO
        MU   = (VS**2)/BRHO
        RO   = BRHO
                
        return BRHO, LAM, MU, RO
#==============================================================================

#==============================================================================    
    def geramdamp(self,test,VP):
    
        nptx   = test.nptx
        npty   = test.npty
        nptz   = test.nptz
        
        npmlx  = test.npmlx
        npmly  = test.npmly
        npmlz  = test.npmlz
        
        X0grid = test.X0grid
        Y0grid = test.Y0grid
        Z0grid = test.Z0grid
        
        vpmax  = np.amax(VP)
        D0   = self.fdamp_will(X0grid,Y0grid,Z0grid,vpmax,test)
        D0   = np.transpose(D0, (1, 0, 2))
         
        D0[npmlx:-npmlx,npmly:-npmly,npmlz:-npmlz] = 0.
        
        return D0
#==============================================================================

#==============================================================================
    def fdamp_will(self,x,y,z,vmax,test):
        vp = vmax
        R = 1e-3   # Reflection coefficient
        
        deltax  = test.deltax
        deltay  = test.deltay 
        deltaz  = test.deltaz 
        
        x0pml   = test.x0pml
        x1pml   = test.x1pml
        
        y0pml   = test.y0pml
        y1pml   = test.y1pml
        
        z0pml   = test.z0pml
        z1pml   = test.z1pml
        
        Lx = deltax
        Ly = deltay
        Lz = deltaz
        
        a = np.where(x<=x0pml, np.abs(x-x0pml), np.where(x>=x1pml, np.abs(x-x1pml), 0.))
        b = np.where(y<=y0pml, np.abs(y-y0pml), np.where(y>=y1pml, np.abs(y-y1pml), 0.))
        c = np.where(z<=z0pml, np.abs(z-z0pml), np.where(z>=z1pml, np.abs(z-z1pml), 0.))
        
        adamp = np.log(1/R)*(3*vp/(2*Lx))*(a/Lx)**2
        bdamp = np.log(1/R)*(3*vp/(2*Ly))*(b/Ly)**2
        cdamp = np.log(1/R)*(3*vp/(2*Lz))*(c/Lz)**2
        
        fdamp = (adamp + bdamp + cdamp)
        
        return fdamp
#==============================================================================

#==============================================================================    
def gerapesos(self,test):
    
    nptx     = test.nptx
    npty     = test.npty
    nptz     = test.nptz

    npmlx    = test.npmlx
    npmly    = test.npmly
    npmlz    = test.npmlz

    pesosx   = np.zeros(npmlx)
    pesosy   = np.zeros(npmly)
    pesosz   = np.zeros(npmlz)

    habcw    = 1

    if(habcw==1):

        for i in range(0,npmlx):
            pesosx[i] = (npmlx-i)/(npmlx)
        
        for i in range(0,npmly):
            pesosy[i] = (npmly-i)/(npmly)
        
        for i in range(0,npmlz):
            pesosz[i] = (npmlz-i)/(npmlz)
        
    if(habcw==2):
       
        mx      = 2
        my      = 2
        mz      = 2
        alphax  = 1.5 + 0.07*(npmlx-mx)    
        alphay  = 1.5 + 0.07*(npmly-my)
        alphaz  = 1.5 + 0.07*(npmlz-mz)

        for i in range(0,npmlx):
            
            if(0<=i<=(mx)):
                pesosx[i] = 1
            elif((mx+1)<=i<=npmlx-1):
                pesosx[i] = ((npmlx-i)/(npmlx-mx))**(alphax)
            else:
                pesosx[i] = 0

        for i in range(0,npmly):
            if(0<=i<=(my)):
                pesosy[i] = 1
            elif((my+1)<=i<=npmly-1):
                pesosy[i] = ((npmly-i)/(npmly-my))**(alphay)
            else:
                pesosy[i] = 0
           
        for i in range(0,npmlz):
               if(0<=i<=(mz)):
                   pesosz[i] = 1
               elif((mz+1)<=i<=npmlz-1):
                   pesosz[i] = ((npmlz-i)/(npmlz-mz))**(alphaz)
               else:
                   pesosz[i] = 0
           
    Mpesosx = np.zeros((nptx,npty,nptz))
    Mpesosy = np.zeros((nptx,npty,nptz))
    Mpesosz = np.zeros((nptx,npty,nptz))
    
    for k in range(0,npmlx):
            
        ai = k
        af = nptx - k - 1 
        bi = 0
        bf = npty #-k
        ci = 0
        cf = nptz #-k
    
        Mpesosx[ai,bi:bf,ci:cf] = pesosx[k]
        Mpesosx[af,bi:bf,ci:cf] = pesosx[k]
                        
    for k in range(0,npmly):
            
        ai = 0
        af = nptx #-k 
        bi = k        
        bf = npty - k - 1        
        ci = 0
        cf = nptz #-k
    
        Mpesosy[ai:af,bi,ci:cf] = pesosy[k]
        Mpesosy[ai:af,bf,ci:cf] = pesosy[k]
    
    for k in range(0,npmlz):
            
        ai = 0
        af = nptx #-k 
        bi = 0        
        bf = npty #-k       
        ci = k
        cf = nptz - k - 1
    
        Mpesosz[ai:af,bi:bf,ci] = pesosz[k]
        Mpesosz[ai:af,bi:bf,cf] = pesosz[k]

    return Mpesosx,Mpesosy,Mpesosz
#==============================================================================
