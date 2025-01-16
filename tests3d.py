#==============================================================================
import math
import numpy as np
from examples.seismic.source import Receiver, RickerSource
import pickle
#==============================================================================

#==============================================================================
class Test01:
#==============================================================================
    def __init__(self):       
#==============================================================================
        ##
        # Grid definition
        #
        self.nx = 101
        self.ny = 111
        self.nz = 121

        # Model cells sizes (m)
        dx = 5.0
        dy = dx 
        dz = dx
    
        # Model size
        Lx = (self.nx-1)*dx 
        Ly = (self.ny-1)*dy
        Lz = (self.nz-1)*dz

        # Grid spacing
        self.hxv   = dx
        self.hyv   = dy
        self.hzv   = dz
        
        # Grid limits
        self.x0vel = 0.
        self.x1vel = Lx
        self.compx = self.x1vel - self.x0vel
        
        self.y0vel = 0.   
        self.y1vel = Ly
        self.compy = self.y1vel - self.y0vel    
        
        self.z0vel = 0. 
        self.z1vel = Lz
        self.compz = self.z1vel-self.z0vel
        
        self.nptx  = self.nx
        self.npty  = self.ny
        self.nptz  = self.nz
        
        # Absorbing boundary grid dimensions
        self.npmlx  = 10
        self.npmly  = 10
        self.npmlz  = 10
        
        self.deltax = self.npmlx*self.hxv
        self.deltay = self.npmly*self.hyv
        self.deltaz = self.npmlz*self.hzv
        
        # Adjust dimensions
        self.x0    = self.x0vel - self.deltax
        self.x1    = self.x1vel + self.deltax
        self.compx = self.compx + 2*self.deltax
        self.nptx  = self.nptx  + 2*self.npmlx 
        
        self.y0    = self.y0vel - self.deltay
        self.y1    = self.y1vel + self.deltay
        self.compy = self.compy + 2*self.deltay
        self.npty  = self.npty  + 2*self.npmly
        
        self.z0    = self.z0vel - self.deltaz
        self.z1    = self.z1vel + self.deltaz
        self.compz = self.compz + 2*self.deltaz
        self.nptz  = self.nptz  + 2*self.npmlz

        self.x0pml = self.x0 + self.npmlx*self.hxv 
        self.x1pml = self.x1 - self.npmlx*self.hxv 
        
        self.y0pml = self.y0 + self.npmly*self.hyv 
        self.y1pml = self.y1 - self.npmly*self.hyv 
        
        self.z0pml = self.z0 + self.npmlz*self.hzv
        self.z1pml = self.z1 - self.npmlz*self.hzv 
        
        X0     = np.linspace(self.x0,self.x1,self.nptx) 
        Y0     = np.linspace(self.y0,self.y1,self.npty) 
        Z0     = np.linspace(self.z0,self.z1,self.nptz) 
        self.X0grid,self.Y0grid,self.Z0grid = np.meshgrid(X0,Y0,Z0)

        # FD simulation parameters
        self.so = 8          # Space order
        self.to = 1          # Time order
        self.t0 = 0.0        # Start time
        self.tn = 1000.      # End time (ms)
        self.f0 = 0.030      # Source (Ricker) peak frequency
        
        # Source location
        self.xsource = Lx/2 
        self.ysource = Ly/2
        self.zsource = Lz/2

        self.rx_x = self.xsource + 200
        self.rx_y = self.ysource 
        self.rx_z = self.zsource
#==============================================================================

#==============================================================================
# Physical model
#==============================================================================

#==============================================================================    
    def get_model(self):
        Vp = 3000.            # m/s
        Vs = Vp/math.sqrt(3)  # m/s
        self.rho = 1500.      # kg/m3
        
        model_shape = (self.nx, self.ny, self.nz)
        
        m_vp = Vp*np.ones(model_shape) 
        m_vs = Vs*np.ones(model_shape) 
        m_rho = self.rho*np.ones(model_shape)         
       
        return m_vp, m_vs, m_rho
#==============================================================================

#==============================================================================
    def build_source(self, src, v, sigma, dt, dt0, f0):
        """
        """        
        src.coordinates.data[:] = [self.xsource, self.ysource, self.zsource]

        den = self.hxv*self.hyv*self.hzv
        src_vz = src.inject(field=v.forward[2], expr=dt*src/den/self.rho)
        
        return [src_vz]
#==============================================================================

#==============================================================================
    def build_receivers(self, grid, time_range, v, sigma):
        self.rec_list = []
        
        nrec = 1
        
        rec_vx = Receiver(name="rec_vx", grid=grid, npoint=nrec, time_range=time_range)
        rec_vx.coordinates.data[:, 0] = self.rx_x
        rec_vx.coordinates.data[:, 1] = self.rx_y
        rec_vx.coordinates.data[:, 2] = self.rx_z

        self.rec_list.append({"name":"rec_vx", "object":rec_vx})
        rec_term_vx = rec_vx.interpolate(expr=v[0])

        return [rec_term_vx]
#==============================================================================

#==============================================================================
    def post_process(self, v, sigma):

        npmlx = self.npmlx
        npmly = self.npmly
        npmlz = self.npmlz          

        #Vx 
        vx_field = v[0].data[0]
        fname = 'files/vx_snap3d.npy'
        np.save(fname, vx_field)

        #Vy
        vy_field = v[1].data[0]
        fname = 'files/vy_snap3d.npy'
        np.save(fname, vy_field)

        #Vz
        vz_field = v[2].data[0]
        fname = 'files/vz_snap3d.npy'
        np.save(fname, vz_field)
#==============================================================================

#==============================================================================
    def post_process2(self, energy):
        
        npmlx = self.npmlx
        npmly = self.npmly
        npmlz = self.npmlz          

        #Energy
        
        enegry_field = energy.data[:]
        fname        = 'files/energy.npy'
                
        with open(fname, 'wb') as f:

            pickle.dump(enegry_field, f, protocol=4)

        f.close()
#==============================================================================

#==============================================================================
test = Test01()
#==============================================================================
