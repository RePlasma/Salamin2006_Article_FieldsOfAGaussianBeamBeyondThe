#!/usr/bin/python
"""
    # Fields of a Gaussian beam beyond the paraxial approximation
    Y. Salamin Appl. Phys. B 86, 319–326 (2007)
    https://link.springer.com/article/10.1007/s00340-006-2442-4
"""

# numpy
import numpy as np
np.random.seed(19680801)
from numpy.random import default_rng
rng = default_rng()
# import functions
from scipy.special import kv, iv, erf
from scipy.integrate import quad
from numpy import log, log10, sin, cos, exp, sqrt, pi, arctan
# interpolate
from scipy import interpolate
# root finding
from scipy.optimize import fsolve
from scipy import optimize
# plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

################################################################
# Salamin 2007 Tight-Focusing field expressions
################################################################

def ExyzBxyz(x, y, z, lbd, W0, E0):
    """
        [Salamin2007]
        eq A1-A6
        field components up to order eps**10
    """
    
    c = 1 # set c=1 in these units 
    
    zR = pi*W0**2/lbd; #[\mu m] Rayleigh range
    eps = W0/zR; #[] epsilon, diffraction angle
    
    # reduced variables
    xi = x / W0; #[] eq (6)
    nu = y / W0; #[] eq (6)
    zeta = z / zR; #[] eq (6)
    r = sqrt(x**2 + y**2) #[\mu m]
    rho = sqrt(xi**2 + nu**2) #[] eq (6)
    
    # spotsize(z)
    W = W0 * sqrt(1+zeta**2)
        
    # phase eq(32)
    psi0 = pi/2 # set psi0=0
    w = 2*pi*c/lbd # laser frequency (but we will not use this value)
    k = 2*pi/lbd #[\mu m^-1]
    t = 0 # set t=0
    
    # regularize R(z) when z=0 eq(32)
    if(np.abs(z)>1e-5):
        psiR = 0.5*k * r**2/(z + zR**2/z)
    else:
        psiR = 0
    psi = psi0 + w*t - k*z - psiR 
    
    # Guoy phase
    psiG = arctan(zeta); #[] eq (14)
    #f = exp(1j*psiG)/sqrt(1+zeta**2); #[] eq (14)
    
    # auxialiary functions
    def Cn(n):
        return (W0/W)**n * cos(psi + n*psiG)
    def Sn(n):
        return (W0/W)**n * sin(psi + n*psiG)
    C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18 = Cn(0), Cn(1), Cn(2), Cn(3), Cn(4), Cn(5), Cn(6), Cn(7), Cn(8), Cn(9), Cn(10), Cn(11), Cn(12), Cn(13), Cn(14), Cn(15), Cn(16), Cn(17), Cn(18)
    S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18 = Sn(0), Sn(1), Sn(2), Sn(3), Sn(4), Sn(5), Sn(6), Sn(7), Sn(8), Sn(9), Sn(10), Sn(11), Sn(12), Sn(13), Sn(14), Sn(15), Sn(16), Sn(17), Sn(18)
    
    # equation 29
    Ee = E0*exp(-(r/W)**2)
    
    # eq A-1 done
    Ex = Ee*(
        S1 + \
        eps**2 * (xi**2 * S3 - rho**4 * S4/4) + \
        eps**4 * (S3/8 - rho**2 * S4/4 - rho**2 * (rho**2 - 16*xi**2)*S5/16 - rho**4* (rho**2 + 2*xi**2)*S6/8 + rho**8*S7/32) + \
        eps**6 * (3*S4/16 - 3/16*rho**2*S5 - 9/32*rho**4*S6 + 15/16*rho**4*xi**2*S7 - rho**6*(rho**2+6*xi**2)*S8/16 + rho**8*(rho**2+xi**2)*S9/32 - rho**12*S10/384) + \
        eps**8 * (9*S5/32 - 3*S6*rho**2/16 - 21*S7*rho**4/64 - 3*S8*rho**6/16 + S9*(7*xi**2*rho**6/8 + 11*rho**8/256) + S10*(-7*xi**2*rho**8/16 - 5*rho**10/128) + S11*(xi**2*rho**10/16 + 13*rho**12/512) + S12*(-xi**2*rho**12/384-rho**14/256) +S13*rho**16/6144 ) + \
        eps**10 * (15*S6/32 - 15*rho**2*S7/64 - 15*rho**4*S8/32 - 35*rho**6*S9/128 - 25*rho**8*S10/256 + S11*(105*xi**2*rho**8/128+29*rho**10/512) + S12*(-15*xi**2*rho**10/32 - 47*rho**12/1536) + S13*(45*xi**2*rho**12/512 + 31*rho**14/1536) + S14*(-5*xi**2*rho**14/768 - 13*rho**16/3072) + S15*(xi**2*rho**16/6144 + rho**18/3072) - S16*rho**20/122880 )
    )
    
    # eq A-2 done
    Ey = Ee*xi*nu*(
        eps**2 * S3 + \
        eps**4 * (rho**2*S5 - rho**4*S6/4) + \
        eps**6 * (15*rho**4*S7/16 - 3*rho**6*S8/8 + rho**8*S9/32) + \
        eps**8 * (7*rho**6*S9/8 - 7*rho**8*S10/16 + rho**10*S11/16 - rho**12*S12/384) + \
        eps**10 * (105*rho**8*S11/128 - 15*rho**10*S12/32 + 45*rho**12*S13/512 - 5*rho**14*S14/768 + rho**16*S15/6144)    
    )
    
    # eq A-3 done
    Ez = Ee*xi*(
        eps*C2 + \
        eps**3 * (-C3/2 + rho**2*C4 - rho**4*C5/4) + \
        eps**5 * (-3*C4/8 - 3*rho**2*C5/8 + 17*rho**4*C6/16 - 3*rho**6*C7/8 + rho**8*C8/32) + \
        eps**7 * (-3*C5/8 - 3*rho**2*C6/8 - 3*rho**4*C7/16 + 33*rho**6*C8/32 - 29*rho**8*C9/64 + rho**10*C10/16 - rho**12*C11/384) + \
        eps**9 * (-15*C6/32 - 15*rho**2*C7/32 - 15*rho**4*C8/64 - 5*rho**6*C9/64 + 247*rho**8*C10/256 - 127*rho**10*C11/256 + 137*rho**12*C12/1536 - 5*rho**14*C13/768 + rho**16*C14/6144 ) + \
        eps**11 * (-45*C7/64 - 45*rho**2*C8/64 - 45*rho**4*C9/128 - 15*C10*rho**6/128 - 15*rho**8*C11/512 + 459*rho**10*C12/512 - 529*rho**12*C13/1024 + 113*C14*rho**14/1024 - 133*C15*rho**16/12288 + rho**18*C16/2048 - rho**20*C17/122880) \
    )
    
    # eq A-4
    Bx = 0
   
    # eq A-5 done
    By = Ee/c * (
        S1 + eps**2 * (rho**2*S3/2 - rho**4*S4/4) + \
        + eps**4 * (-S3/8 + rho**2*S4/4 + 5*rho**4*S5/16 - rho**6*S6/4 + rho**8*S7/32) + \
        + eps**6 * (-3*S4/16 + 3*rho**2*S5/16 + 9*rho**4*S6/32 + 5*rho**6*S7/32 - 7*rho**8*S8/32 + 3*rho**10*S9/64 - rho**12*S10/384) + \
        + eps**8 * (-9*S5/32 + 3*rho**2*S6/16 + 21*rho**4*S7/64 + 3*rho**6*S8/16 + 17*rho**8*S9/256 - 23*rho**10*S10/128 + 27*rho**12*S11/512 - rho**14*S12/192 + rho**16*S13/6144) + \
        + eps**10 * (-15*S6/32 + 15*rho**2*S7/64 + 15*rho**4*S8/32 + 35*rho**6*S9/128 + 25*rho**8*S10/256 + 13*rho**10*S11/512 - 223*rho**12*S12/1536 + 163*rho**14*S13/3072 - 11*rho**16*S14/1536 + 5*rho**18*S15/12288 - rho**20*S16/122880) \
    )
    
    # eq A-6 done
    Bz = Ee/c*nu * (
        eps * (C2) + \
        eps**3 * (C3/2 + rho**2*C4/2 - rho**4*C5/4) + \
        eps**5 * (3*C4/8 + 3*rho**2*C5/8 + 3*rho**4*C6/16 - rho**6*C7/4 + rho**8*C8/32) + \
        eps**7 * (3*C5/8 + 3*rho**2*C6/8 + 3*rho**4*C7/16 + rho**6*C8/16 - 13*rho**8*C9/64 + 3*rho**10*C10/64 - rho**12*C11/384) + \
        eps**9 * (15*C6/32 + 15*rho**2*C7/32 + 15*rho**4*C8/64 + 5*rho**6*C9/64 + 5*rho**8*C10/256 - 41*rho**10*C11/256 + 79*rho**12*C12/1536 - rho**14*C13/192 + rho**16*C4/6144) + \
        eps**11 * (45*C7/64 - 131*C13*rho**12/1024 + 13*C14*rho**14/256 - 29*C15*rho**16/4096 + 5*C16*rho**18/12288 - C17*rho**20/122880) \
    )
    
    return Ex, Ey, Ez, Bx, By, Bz

def kvec(x, y, z, lbd, W0, E0):
    """
        Pointing vector
    """
    Evec = [0,0,0]
    Bvec = [0,0,0]
    Evec[0], Evec[1], Evec[2], Bvec[0], Bvec[1], Bvec[2] = ExyzBxyz(x, y, z, lbd, W0, E0)
    return np.cross(Evec,Bvec)

def ExFocalPlane0(x, y, z, lbd, W0, E0):
    """
    eq B1
    """
    
    c = 1 # set c=1 in these units 
    
    zR = pi*W0**2/lbd; #[\mu m] 
    eps = W0/zR; #[] epsilon, diffraction angle
    
    # reduced variables
    xi = x / W0; #[] eq (6)
    nu = y / W0; #[] eq (6)
    zeta = z / zR; #[] eq (6)
    rho = sqrt(xi**2 + nu**2) #[] eq (6)
    
    # phase eq(32)
    psi0 = pi/2 # set psi0=pi/2
    w = 2*pi*c/lbd # laser frequency (but we will not use this value)
    k = 2*pi/lbd #[\mu m^-1]
    t = 0 # set t=0
    
    # regularize R(z) when z=0 eq(32)
    if(np.abs(z)>1e-5):
        psiR = 0.5*k * r**2/(z + zR**2/z)
    else:
        psiR = 0
    psi = psi0 + w*t - k*z - psiR 
    
    # Guoy phase
    psiG = arctan(zeta); #[] eq (14)
    
    return E0*exp(-rho**2)*( 1 ) * sin(psi0+w*t)

def ExFocalPlane4(x, y, z, lbd, W0, E0):
    """
    eq B1
    """
    
    c = 1 # set c=1 in these units 
    
    zR = pi*W0**2/lbd; #[\mu m] 
    eps = W0/zR; #[] epsilon, diffraction angle
    
    # reduced variables
    xi = x / W0; #[] eq (6)
    nu = y / W0; #[] eq (6)
    zeta = z / zR; #[] eq (6)
    rho = sqrt(xi**2 + nu**2) #[] eq (6)
    
    # phase eq(32)
    psi0 = pi/2 # set psi0=pi/2
    w = 2*pi*c/lbd # laser frequency (but we will not use this value)
    k = 2*pi/lbd #[\mu m^-1]
    t = 0 # set t=0
    
    # regularize R(z) when z=0 eq(32)
    if(np.abs(z)>1e-5):
        psiR = 0.5*k * r**2/(z + zR**2/z)
    else:
        psiR = 0
    psi = psi0 + w*t - k*z - psiR 
    
    # Guoy phase
    psiG = arctan(zeta); #[] eq (14)
    
    return E0*exp(-rho**2)*( 1 + eps**2*(xi**2-rho**4/4) + \
                               + eps**4*(1/8 - rho**2/4 + xi**2*rho**2 - rho**4/16 - xi**2*rho**4/4 - rho**6/8 + rho**8/32) ) * sin(psi0+w*t)

def ExFocalPlane10(x, y, z, lbd, W0, E0):
    """
    eq B1
    """
    
    c = 1 # set c=1 in these units 
    
    zR = pi*W0**2/lbd; #[\mu m] 
    eps = W0/zR; #[] epsilon, diffraction angle
    
    # reduced variables
    xi = x / W0; #[] eq (6)
    nu = y / W0; #[] eq (6)
    zeta = z / zR; #[] eq (6)
    rho = sqrt(xi**2 + nu**2) #[] eq (6)
    
    # phase eq(32)
    psi0 = pi/2 # set psi0=pi/2
    w = 2*pi*c/lbd # laser frequency (but we will not use this value)
    k = 2*pi/lbd #[\mu m^-1]
    t = 0 # set t=0
    
    # regularize R(z) when z=0 eq(32)
    if(np.abs(z)>1e-5):
        psiR = 0.5*k * r**2/(z + zR**2/z)
    else:
        psiR = 0
    psi = psi0 + w*t - k*z - psiR 
    
    # Guoy phase
    psiG = arctan(zeta); #[] eq (14)
    
    return E0*exp(-rho**2)*( 1 + eps**2*(xi**2-rho**4/4) + \
                               + eps**4*(1/8 - rho**2/4 + xi**2*rho**2 - rho**4/16 - xi**2*rho**4/4 - rho**6/8 + rho**8/32) \
                               + eps**6*( 3/16 - 3*rho**2/16 - 9*rho**4/32 + 15*xi**2*rho**4/16 - 3*xi**2*rho**6/8 - rho**8/16 + xi**2*rho**8/32 + rho**10/32 - rho**12/384 ) \
                               + eps**8*(9/32 - 3*rho**2/16 - 21*rho**4/64 - 3*rho**6/16 + 7*xi**2*rho**6/8 + 11*rho**8/256 - 7*xi**2*rho**8/16 - 5*rho**10/128 + xi**2*rho**10/16 + 13*rho**12/512 - xi**2*rho**12/384 - rho**14/256 + rho**16/6144)
                               + eps**10*(15/32 - 15*rho**2/64 - 15*rho**4/32 - 35*rho**6/128 - 25*rho**8/256 + (105*xi**2*rho**8/128+29*rho**10/512) + (-15*xi**2*rho**10/32 - 47*rho**12/1536) + (45*xi**2*rho**12/512+31*rho**14/1536) + (-5*xi**2*rho**14/768-13*rho**16/3072) + (xi**2*rho**16/6144 + rho**18/3072) - rho**20/122880 )
                               ) * sin(psi0+w*t)


def EzFocalPlane1(x, y, z, lbd, W0, E0):
    """
    eq B3
    """
    
    c = 1 # set c=1 in these units 
    
    zR = pi*W0**2/lbd; #[\mu m] 
    eps = W0/zR; #[] epsilon, diffraction angle
    
    # reduced variables
    xi = x / W0; #[] eq (6)
    nu = y / W0; #[] eq (6)
    zeta = z / zR; #[] eq (6)
    rho = sqrt(xi**2 + nu**2) #[] eq (6)
    
    # phase eq(32)
    psi0 = pi/2 # set psi0=pi/2
    w = 2*pi*c/lbd # laser frequency (but we will not use this value)
    k = 2*pi/lbd #[\mu m^-1]
    t = 0 # set t=0
    
    # regularize R(z) when z=0 eq(32)
    if(np.abs(z)>1e-5):
        psiR = 0.5*k * r**2/(z + zR**2/z)
    else:
        psiR = 0
    psi = psi0 + w*t - k*z - psiR 
    
    # Guoy phase
    psiG = arctan(zeta); #[] eq (14)
    
    return E0*exp(-rho**2)*xi*( eps ) * cos(psi0+w*t)

def EzFocalPlane5(x, y, z, lbd, W0, E0):
    """
    eq B3
    """
    
    c = 1 # set c=1 in these units 
    
    zR = pi*W0**2/lbd; #[\mu m] 
    eps = W0/zR; #[] epsilon, diffraction angle
    
    # reduced variables
    xi = x / W0; #[] eq (6)
    nu = y / W0; #[] eq (6)
    zeta = z / zR; #[] eq (6)
    rho = sqrt(xi**2 + nu**2) #[] eq (6)
    
    # phase eq(32)
    psi0 = pi/2 # set psi0=pi/2
    w = 2*pi*c/lbd # laser frequency (but we will not use this value)
    k = 2*pi/lbd #[\mu m^-1]
    t = 0 # set t=0
    
    # regularize R(z) when z=0 eq(32)
    if(np.abs(z)>1e-5):
        psiR = 0.5*k * r**2/(z + zR**2/z)
    else:
        psiR = 0
    psi = psi0 + w*t - k*z - psiR 
    
    # Guoy phase
    psiG = arctan(zeta); #[] eq (14)
    
    return E0*exp(-rho**2)*xi*( eps + \
                               + eps**3*(-0.5+rho**2-rho**4/4) \
                               + eps**5*(-3/8-3*rho**2/8+17*rho**4/16-3*rho**6/8+rho**8/32) \
                              ) * cos(psi0+w*t)

def EzFocalPlane11(x, y, z, lbd, W0, E0):
    """
    eq B3
    """
    
    c = 1 # set c=1 in these units 
    
    zR = pi*W0**2/lbd; #[\mu m] 
    eps = W0/zR; #[] epsilon, diffraction angle
    
    # reduced variables
    xi = x / W0; #[] eq (6)
    nu = y / W0; #[] eq (6)
    zeta = z / zR; #[] eq (6)
    rho = sqrt(xi**2 + nu**2) #[] eq (6)
    
    # phase eq(32)
    psi0 = pi/2 # set psi0=pi/2
    w = 2*pi*c/lbd # laser frequency (but we will not use this value)
    k = 2*pi/lbd #[\mu m^-1]
    t = 0 # set t=0
    
    # regularize R(z) when z=0 eq(32)
    if(np.abs(z)>1e-5):
        psiR = 0.5*k * r**2/(z + zR**2/z)
    else:
        psiR = 0
    psi = psi0 + w*t - k*z - psiR 
    
    # Guoy phase
    psiG = arctan(zeta); #[] eq (14)
    
    return E0*exp(-rho**2)*xi*( eps + \
                               + eps**3*(-0.5+rho**2-rho**4/4) \
                               + eps**5*(-3/8-3*rho**2/8+17*rho**4/16-3*rho**6/8+rho**8/32) \
                               + eps**7*(-3/8 -3*rho**2/8 - 3*rho**4/16 + 33*rho**6/32 - 29*rho**8/64 + rho**10/16 - rho**12/384) \
                               + eps**9*(-15/32 - 15*rho**2/32 - 15*rho**4/64 - 5*rho**6/64 + 247*rho**8/256 - 127*rho**10/256 + 137*rho**12/1536 - 5*rho**14/768 + rho**16/6144) \
                               + eps**11*(-45/64 - 45*rho**2/64 - 45*rho**4/128 - 15*rho**6/128 - 15*rho**8/512 + 459*rho**10/512 - 529*rho**12/1024 + 113*rho**14/1024 - 133*rho**16/12288 + rho**18/2048 - rho**20/122880)
                              ) * cos(psi0+w*t)
