#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy                   as np
import sys
import math                    as mt
from sympy                     import finite_diff_weights
import time as tm
#==============================================================================

#==============================================================================
# Devito Imports
#==============================================================================
from devito import *
from examples.seismic.source import RickerSource, Receiver, TimeAxis
#==============================================================================

#==============================================================================
# Setup Configuration
#==============================================================================
so         = 8
to         = 1
nptx       = 141
npty       = nptx
nptz       = nptx
x0         = -200
x1         = 1200
compx      = x1 - x0
y0         = x0
y1         = x1
compy      = compx
z0         = x0
z1         = x1
compz      = compx
hxv        = (compx)/(nptx-1)
hyv        = hxv
hzv        = hxv
t0         = 0
tn         = 1000
npmlx      = 20
npmly      = npmlx
npmlz      = npmlx
#==============================================================================

#==============================================================================
# Set Type Test
#==============================================================================
MVT = np.random.rand(nptx,npty,nptz)
#==============================================================================

#==============================================================================
# Definição de Vetores Devito
#==============================================================================
origin         = (x0,y0,z0)       
extent         = (compx,compy,compz)
shape          = (nptx,npty,nptz)   
spacing        = (hxv,hyv,hzv)     
#==============================================================================

#==============================================================================
class dfulldomain(SubDomain):

    name = 'dfull'

    def define(self, dimensions):

        x, y, z = dimensions

        return {x: x, y: y, z:z}
#==============================================================================

#==============================================================================
class d0domain(SubDomain):

    name = 'd0'

    def define(self, dimensions):

        x, y, z = dimensions

        return {x: ('middle', npmlx, npmlx), y: ('middle', npmly, npmly), z: ('middle', npmlz, npmlz)}
#==============================================================================

#==============================================================================
class d1domain(SubDomain):

    name = 'd1'

    def define(self, dimensions):

        x, y, z = dimensions

        return {x: ('left',npmlx), y: y, z: z}
#==============================================================================

#==============================================================================
class d2domain(SubDomain):

    name = 'd2'

    def define(self, dimensions):

        x, y, z = dimensions

        return {x: ('right',npmlx), y: y, z: z}
#==============================================================================

#==============================================================================    
class d3domain(SubDomain):

    name = 'd3'

    def define(self, dimensions):

        x, y, z = dimensions

        return {x: ('middle', npmlx, npmlx), y: ('left', npmly), z: z}
#==============================================================================    

#==============================================================================    
class d4domain(SubDomain):

    name = 'd4'

    def define(self, dimensions):

        x, y, z = dimensions

        return {x: ('middle', npmlx, npmlx), y: ('right',npmly), z: z}
#============================================================================== 

#==============================================================================    
class d5domain(SubDomain):

    name = 'd5'

    def define(self, dimensions):

        x, y, z = dimensions

        return {x: ('middle', npmlx, npmlx), y: ('middle', npmly, npmly), z: ('left',npmlz)}
#==============================================================================    

#==============================================================================    
class d6domain(SubDomain):

    name = 'd6'

    def define(self, dimensions):

        x, y, z = dimensions

        return {x: ('middle', npmlx, npmlx), y: ('middle', npmly, npmly), z: ('right',npmlz)}
#============================================================================== 

#==============================================================================
dfull_domain = dfulldomain()
d0_domain    = d0domain()
d1_domain    = d1domain()
d2_domain    = d2domain()
d3_domain    = d3domain()
d4_domain    = d4domain()
d5_domain    = d5domain()
d6_domain    = d6domain()
grid         = Grid(origin=origin,extent=extent,shape=shape,subdomains=(dfull_domain,d0_domain,d1_domain,d2_domain,d3_domain,d4_domain,d5_domain,d6_domain))
#==============================================================================

#==============================================================================
# Symbolic Dimensions
#==============================================================================
(hx,hy,hz) = grid.spacing_map  
(x, y, z)  = grid.dimensions    
time       = grid.time_dim     
t          = grid.stepping_dim 
dt         = grid.stepping_dim.spacing
#==============================================================================

#==============================================================================
# Lame Parameters 
#==============================================================================
VP   = Function(name="VP",grid=grid,space_order=2)
VS   = Function(name="VS",grid=grid,space_order=2)
RHO  = Function(name="RHO",grid=grid,space_order=2)
LAM  = Function(name="LAM",grid=grid,space_order=2)
MU   = Function(name="MU",grid=grid,space_order=2)
RO   = Function(name="RO",grid=grid,space_order=2)

VP.data[:,:,:]    = MVT
VS.data[:,:,:]    = MVT
RHO.data[:,:,:]   = MVT
LAM.data[:,:,:]   = MVT
MU.data[:,:,:]    = MVT
RO.data[:,:,:]    = MVT
#==============================================================================

#==============================================================================
# Time Construction
#==============================================================================
if(to==1):

    coeffs   = finite_diff_weights(1, range(-so//2+1, so//2+1),0.5)
    cte_temp = np.sum(np.abs(coeffs[-1][-1])) / 2
    cte_cfl  = np.sqrt(grid.dim) / grid.dim / cte_temp
else:
    
    a1      = 4.0
    coeffs  = finite_diff_weights(2, range(-so,so+1), 0)[-1][-1]
    cte_cfl = np.sqrt(a1/np.float(grid.dim*np.sum(np.abs(coeffs))))

vpmax      = np.around(np.amax(MVT),1)
dt_scale   = 0.4
dtmax      = dt_scale*((min(hxv,hyv,hzv)*cte_cfl)/(vpmax))
ntmax      = 1000#int(1e-3*(tn-t0)/dtmax)
dt0        = 1e-3*(tn-t0)/(ntmax)
time_range = TimeAxis(start=t0,stop=tn,num=ntmax+1)
nt         = time_range.num - 1
cur        = (vpmax*dt0)/(min(hxv,hyv,hzv))
nplot      = 25
jump       = mt.ceil(nt/nplot) + 1
#==============================================================================

#==============================================================================
# Weights
#==============================================================================
pesosx                  = Function(name='pesosx', grid=grid, space_order=2)
pesosy                  = Function(name='pesosy', grid=grid, space_order=2)
pesosz                  = Function(name='pesosz', grid=grid, space_order=2)
pesosx.data[:]          = MVT[:]
pesosy.data[:]          = MVT[:]
pesosz.data[:]          = MVT[:]
#==============================================================================

#==============================================================================
# Symbolic Fields Construction
#==============================================================================
v      = VectorTimeFunction(name='v', grid=grid, space_order=so, time_order=1)
tau    = TensorTimeFunction(name='t', grid=grid, space_order=so, time_order=1)
#==============================================================================

#==============================================================================
# Symbolic Equation Construction
#==============================================================================
pde_v = RHO*v.dt - div(tau)
e = 0.5*(grad(v.forward) + grad(v.forward).transpose(inner=False)) 
tr_e = e[0,0] + e[1,1] + e[2,2]
pde_tau = tau.dt - LAM*diag(tr_e) - 2*MU*e 
u_v = Eq(v.forward, solve(pde_v, v.forward),subdomain = grid.subdomains['dfull'])
u_tau = Eq(tau.forward, solve(pde_tau, tau.forward),subdomain = grid.subdomains['dfull'])
#==============================================================================

#==============================================================================
# Higdon Condition - Velocity
#==============================================================================
eqhabchigdon = []

beta  = (1+(VP[x,y,z]/VS[x,y,z]))/(2)

ax    =  hx + (dt/beta)*VP[x,y,z]
bx    =  hx - (dt/beta)*VP[x,y,z]
cx    =  hx + (dt/beta)*VP[x,y,z]
dx    = -hx + (dt/beta)*VP[x,y,z]

ay    =  hy + (dt/beta)*VP[x,y,z]
by    =  hy - (dt/beta)*VP[x,y,z]
cy    =  hy + (dt/beta)*VP[x,y,z]
dy    = -hy + (dt/beta)*VP[x,y,z]

az    =  hz + (dt/beta)*VP[x,y,z]
bz    =  hz - (dt/beta)*VP[x,y,z]
cz    =  hz + (dt/beta)*VP[x,y,z]
dz    = -hz + (dt/beta)*VP[x,y,z]

dn    = ['d1','d3','d5']
dp    = ['d2','d4','d6']

for i in range(0,3):
    
    if(i==0): vpesos = pesosx[x,y,z]
    if(i==1): vpesos = pesosy[x,y,z]
    if(i==2): vpesos = pesosz[x,y,z]

    for k1 in range(0,3):
    
        eqv1 = Eq(v[i][t+1,x,y,z], (1-vpesos)*v[i][t+1,x,y,z] + vpesos*(1/3)*((bx/ax + by/ay + bz/az)*v[i][t,x,y,z] + (cx/ax)*v[i][t,x+1,y,z] + (cy/ay)*v[i][t,x,y+1,z] + (cz/az)*v[i][t,x,y,z+1]
                + (dx/ax)*v[i][t+1,x+1,y,z] + (dy/ay)*v[i][t+1,x,y+1,z] + (dz/az)*v[i][t+1,x,y,z+1]),subdomain = grid.subdomains[dn[k1]])

        eqv2 = Eq(v[i][t+1,x,y,z], (1-vpesos)*v[i][t+1,x,y,z] + vpesos*(1/3)*((bx/ax + by/ay + bz/az)*v[i][t,x,y,z] + (cx/ax)*v[i][t,x-1,y,z] + (cy/ay)*v[i][t,x,y-1,z] + (cz/az)*v[i][t,x,y,z-1]
                + (dx/ax)*v[i][t+1,x-1,y,z] + (dy/ay)*v[i][t+1,x,y-1,z] + (dz/az)*v[i][t+1,x,y,z-1]),subdomain = grid.subdomains[dp[k1]])
        
        eqhabchigdon.append(eqv1)
        eqhabchigdon.append(eqv2)
        
    for j in range(0,3):
            
        for k1 in range(0,3):
                
            eqtau1 = Eq(tau[i,j][t+1,x,y,z], (1-vpesos)*tau[i,j][t+1,x,y,z] + vpesos*(1/3)*((bx/ax + by/ay + bz/az)*tau[i,j][t,x,y,z] + (cx/ax)*tau[i,j][t,x+1,y,z] + (cy/ay)*tau[i,j][t,x,y+1,z] + (cz/az)*tau[i,j][t,x,y,z+1]
                      + (dx/ax)*tau[i,j][t+1,x+1,y,z] + (dy/ay)*tau[i,j][t+1,x,y+1,z] + (dz/az)*tau[i,j][t+1,x,y,z+1]),subdomain = grid.subdomains[dn[k1]])
            
            eqtau2 = Eq(tau[i,j][t+1,x,y,z], (1-vpesos)*tau[i,j][t+1,x,y,z] + vpesos*(1/3)*((bx/ax + by/ay + bz/az)*tau[i,j][t,x,y,z] + (cx/ax)*tau[i,j][t,x-1,y,z] + (cy/ay)*tau[i,j][t,x,y-1,z] + (cz/az)*tau[i,j][t,x,y,z-1]
                      + (dx/ax)*tau[i,j][t+1,x-1,y,z] + (dy/ay)*tau[i,j][t+1,x,y-1,z] + (dz/az)*tau[i,j][t+1,x,y,z-1]),subdomain = grid.subdomains[dp[k1]])
      
            eqhabchigdon.append(eqtau1)
            eqhabchigdon.append(eqtau2)
#==============================================================================

#==============================================================================
# Operator Definition
#==============================================================================
op = Operator(eqhabchigdon, subs=grid.spacing_map)
#==============================================================================

#==============================================================================
# Operator Evolution
#==============================================================================
op(dt=dt0,time=10)
#==============================================================================