# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import os.path

plt.close("all")
#plt.style.use(["science", "no-latex"])

t0 = time.time()

def rad(deg):
    return np.deg2rad(deg)


def deg(rad):
    return np.rad2deg(rad)


def larmor_radius(B):
    return m*1000/(q*np.linalg.norm(B))


def particle(_type):
    e = 1.6*10**(-19)
    if _type == "proton" or _type == "p":
        _m = np.float64(1.672*10**(-27))
        _q = e
    elif _type == "electron" or _type == "e":
        _m = 9.1095*10**(-31)
        _q = -e
    else:
        print("'{}' is not recognized as a particle".format(_type))
        _m, _q = np.zeros(2)
    return _m, _q


def rotmat_general(ax1, ax2, ang):
    """
    Function to calculate the rotation matrix about an arbitrary axis.
    
    ax1: usually z axis [0, 0, 1]
    ax2: usually position vector of point (R)
    ang: rotation angle [rad]
    
    """
    rotax = np.cross(ax1, ax2)
    
    rotax = rotax/np.linalg.norm(rotax)
    ux, uy, uz = np.abs(rotax)
    
    a = 1 - np.cos(ang)
    row1 = [np.cos(ang)+ux**2*a, ux*uy*a-uz*np.sin(ang), ux*uz*a+uy*np.sin(ang)]
    row2 = [uy*ux*a+uz*np.sin(ang), np.cos(ang)+uy**2*a, uy*uz*a-ux*np.sin(ang)]
    row3 = [uz*ux*a-uy*np.sin(ang), uz*uy*a+ux*np.sin(ang), np.cos(ang)+uz**2*a]
    
    RR = np.array([row1, row2, row3])
    return RR
    

def rotmat(axis, ang):
    """ Function to generate a rotation matrix about the three principal axis
    """
    if axis == "x":
        return np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
    if axis == "y":
        return np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    if axis == "z":
        return np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])


def f(B, vel):
    """ Function to calculate the variation by the Lorentz force to be used in
    the RK6 algorithm. This is where the physics come in.
    
    Input:
        B (arr): Magnetic field vector [Bx, By, Bz] at current position.
        vel (arr): Velocity vector [vx, vy, vz] at current position.
    
    Output:
        out (arr): The variation f(vi, Bi) of each component.
    
    """
    v_ = np.linalg.norm(vel)    
    gamma = (1-(v_/c)**2)**(-1/2) 
    out = q/(gamma*m)*np.cross(vel, B)
    return out


def legendre(n, m, theta):
    """ Function to return the Legendre polynomial of degree n and order m, 
    given the angle theta. This is used to model the magnetic field based on
    Connery (1993).
    """
    P = np.zeros([3, 3])
    P[1, 0] = np.cos(theta)
    P[1, 1] = np.sin(theta)
    P[2, 0] = (3/2)*(np.cos(theta)**2-1/3)
    P[2, 1] = np.sqrt(3)*np.cos(theta)*np.sin(theta)
    P[2, 2] = (np.sqrt(3)/2)*np.sin(theta)**2
    return P[n, m]


def D_legendre(n, m, theta):
    """ Function to return the derivative of the Legendre polynomials of 
    degree n and order m, given the angle theta. This is used to model the 
    magnetic field based on Connery (1993).
    """
    dP = np.zeros([3, 3])
    dP[1, 0] = -np.sin(theta)
    dP[1, 1] = np.cos(theta)
    dP[2, 0] = -3*np.cos(theta)*np.sin(theta)
    dP[2, 1] = np.sqrt(3)*2*np.cos(theta)
    dP[2, 2] = np.sqrt(3)*np.cos(theta)*np.sin(theta)
    return dP[n, m]


def Bquad(R, a, dipole=False):
    """ Function to return the Uranian magnetic field vector in Cartesian 
    coordinates, based on the model in Connery (1993). This function requires
    the Legendre polynomials to be defined, as well as matrices g and h of 
    the internal Schmidt coefficients from the AH5 model of Herbert (2009).
    
    Input:
        R (arr): Position vector [x, y, z] of the point where the field is 
                 calculated.
        a (float): The equatorial radius of Uranus in metres.
        dipole (bool): Calculate only the dipole component of the field or not.
                       Default: False, which will calculate full quadrupole +
                       dipole field.
    
    Output:
        [Bx, By, Bz] (arr): The magnetic field vector at R.
    
    """
    # converting to spherical coordinates
    x, y, z = R
    r = np.linalg.norm(R)
    theta = np.arccos(z/r)
    
    # making sure to convert to the right angle is a process...
    if x == 0:
        if y > 0:
            phi = np.pi/2
        if y < 0:
            phi = 3*np.pi/2
    else:
        phi = np.arctan(y/x)
        
    if x < 0:
        phi += np.pi
    else:
        if y < 0:
            phi += 2*np.pi

    # defining variables for B-field in spherical coordinates.
    Br = 0
    Btheta = 0
    Bphi = 0
    
    # deciding order to expand sum to, depending on if only dipole or not
    if dipole:
        order = 1
    else:
        order = 2
    
    # performing sum, based on the field model
    for n in range(order+1):
        for m in range(order+1):
            Br += (n+1)*(a/r)**(n+2)*(g[n, m]*np.cos(m*phi)+h[n, m]*\
                   np.sin(m*phi))*legendre(n, m, theta)
            Btheta += (a/r)**(n+2)*(g[n, m]*np.cos(m*phi)+h[n, m]*\
                   np.sin(m*phi))*D_legendre(n, m, theta)
            Bphi += m*(a/r)**(n+2)*(g[n, m]*np.sin(m*phi)-\
                   h[n, m]*np.cos(m*phi))*legendre(n, m, theta)

    Btheta = -Btheta
    Bphi = Bphi/np.sin(theta)

    # converting back to Cartesian coordinates
    Bx = np.sin(theta)*np.cos(phi)*Br + np.cos(theta)*np.cos(phi)*Btheta-np.sin(phi)*Bphi
    By = np.sin(theta)*np.sin(phi)*Br + np.cos(theta)*np.sin(phi)*Btheta + np.cos(phi)*Bphi
    Bz = np.cos(theta)*Br - np.sin(theta)*Btheta
    
    return np.array([Bx, By, Bz])


# defining the matrices of internal Schmidt coefficients from Herbert (2009)
# as g[n, m] and h[n, m]

g = np.zeros([3, 3])
g[1, 0] = 11278
g[1, 1] = 10928
g[2, 0] = -9648
g[2, 1] = -12284
g[2, 2] = 1453

h = np.zeros([3, 3])
h[1, 1] = -16049
h[2, 1] = 6405
h[2, 2] = 4220

# coefficients are in nT
h = h*1e-9
g = g*1e-9

# creating rotation matrices to perform coordinate transform to a
# dipole-aligned coordinate system
g_vec = [g[1, 1], h[1, 1], g[1, 0]]
modg = np.linalg.norm(g_vec)
g_dir = g_vec/modg # this is the direction of the dipole
gx, gy, gz = g_dir
ang1 = np.arctan2(gx, gz)

R1 = rotmat("y", ang1) 
R1_inv = np.linalg.inv(R1)
v1 = np.matmul(R1_inv, g_dir)
ang2 = np.arctan2(v1[1], v1[2])

R2 = rotmat("x", -ang2)
R2_inv = np.linalg.inv(R2)

rotmat2 = np.matmul(R2_inv, R1_inv)
rotmat1 = np.matmul(R1, R2)


# constants
Ru = 25362e3 # equatorial radius of Uranus
c = np.float64(299792458)
t1 = np.sqrt(21) # useful in RK6 method

particleType = "proton"
m, q = particle(particleType)

# parameters
alpha = np.deg2rad(15) # equatorial pitch angle
phang = np.deg2rad(0) # initial gyro-phase

L = 6.5 # L-shell of particle
r0 = L*Ru

mev = 10 # energy of particle, in MeV
Ek = np.float64(mev*1e6*1.6*10**(-19))
v0 = np.float64(c*np.sqrt(1-((m*c**2)/(m*c**2+Ek))**2))

seconds = 10 # "Real" time to run the simulation over

save_vel = False # save velocity history or not
savefile = False # save particle trace or not
is_dipole = True # dipole or not

# setting initial position at magnetic equator [x0, 0, 0]
# (dipole aligned coordinate system)
x0 = r0
y0 = 0
z0 = 0
R0 = [x0, y0, z0]

# finding initial magnetic field, which will determine time-step in RK6
R0_rotated = np.matmul(rotmat1, R0)
B_in_rotated = Bquad(R0_rotated, Ru, dipole=is_dipole)
b_initial = np.matmul(rotmat2, B_in_rotated)
B0 = np.linalg.norm(b_initial)
rL = larmor_radius(B0)
b_dir = b_initial/B0

# important line: deciding timestep size which will determine the accuracy,
# but also the simulation time required. Initially set to 1/50 of the
# gyro-period at the initial position, but reduced by another factor of 10 to
# improve accuracy
dt = np.float64((1/50)*(2*np.pi*m)/(np.abs(q)*B0*10))

# calculating number of iterations needed given desired length and timestep
N = int(round(seconds/dt))

# setting up arrays for position and velocity
V = np.zeros([N, 3])
R = np.zeros([N, 3])
R[0] = R0

# setting up initial velocity, making sure to always get the correct pitch angle
V00 = -v0*b_dir # velocity vector aligned with field direction
rotation1 = rotmat("y", alpha) # rotating a bit to get rotaxes
V0 = np.matmul(rotation1, V00)
rotax1 = V0/v0
rotax2 = b_dir
rotation2 = rotmat("y", -alpha) # rotating back to original pos
V0 = np.matmul(rotation2, V0)
rotation3 = rotmat_general(rotax1, rotax2, alpha) # rotating along correct axis
V0 = np.matmul(rotation3, V0)

V[0] = V0

# main part of the program, where the RK6 method (Luther 1968) is employed to
# solve the EoM for the particle in the magnetic field of Uranus.
for i in range(N-1):
    r = R[i]
    v = V[i]
    
    # 1) rotate position to equivalent in "tilted-field" coordinates
    # 2) calculate magnetic field vector at this point
    # 3) rotate magnetic field vector back to original point
    r_rotated = np.matmul(rotmat1, r)
    B_rotated = Bquad(r_rotated, Ru, dipole=is_dipole)
    B = np.matmul(rotmat2, B_rotated)

    # defining the Runge-Kutta formulas (Luther 1968)
    k1 = dt*f(B, v)
    k2 = dt*f(B, v + k1)
    k3 = dt*f(B, v + (3*k1+k2)/8)
    k4 = dt*f(B, v + (8*k1+2*k2+8*k3)/27)
    k5 = dt*f(B, v + (3*(3*t1-7)*k1-8*(7-t1)*k2+48*(7-t1)*k3-3*(21-t1)*k4)/392)
    k6 = dt*f(B, v + ((-5*(231+51*t1))*k1-40*(7+t1)*k2-320*t1*k3+3*(21+121*t1)*k4+392*(6+t1)*k5)/1960)
    k7 = dt*f(B, v + (15*(22+7*t1)*k1+120*k2+40*(7*t1-5)*k3-63*(3*t1-2)*k4-14*(49+9*t1)*k5+70*(7-t1)*k6)/180)

    # calculating velocity difference
    dv = (9*k1+64*k3+49*k5+49*k6+9*k7)/180
    
    # failsafe, could trigger if timestep is too large for example
    if np.linalg.norm(V[i] + dv) > c:
        print("Value > c encountered: \n i = {} \n R = {} \n V = {} \n dv = {}".format(i, R[i], V[i], dv))
        break
    
    # estimating next position and velocity
    V[i+1] = V[i] + dv
    R[i+1] = R[i] + V[i+1]*dt
    
    # print statements to track progress and estimate simulation time
    if i > 0 and i != 10000:         
        if i % 10000 == 0:
            print("Step {} of {}".format(i, N))
            if np.linalg.norm(r) > 10*L*Ru: # stopping simulation if particle escapes
                print("Particle escaped :(")
                R[i:, :] = np.nan
                break
    if i == 0:
        t00 = time.time()
        print("Starting simulation...")
    if i == 10000:
        tt = time.time() - t00
        estimatedTime = (N/10000)*tt
        print("Estimating it will take {:.1f} s. ({:.1f} min)".format(estimatedTime, estimatedTime/60))
        
t = time.time()
print("DONE!")
print("That took {:.1f} s.".format(t-t0))


# ----- PLOTTING -----
# plotting trajectory of particle around Uranus
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(R[:, 0]/Ru, R[:, 1]/Ru, R[:, 2]/Ru, linewidth=2)
ax.set_xlim([-L-1, L+1])
ax.set_ylim([-L-1, L+1])
ax.set_zlim([-L-1, L+1])
ax.set_xlabel("x [Ru]", fontsize=16)
ax.set_ylabel("y [Ru]", fontsize=16)
ax.set_zlabel("z [Ru]", fontsize=16)
title = r"$E_k = $" + "{}".format(mev) + r"$MeV$" +", " + r"$L = $" + "{}".format(L) + ", " + r"$\alpha_{eq} = $" + "{:.1f}".format(np.rad2deg(alpha))
ax.set_title(title, fontsize=18)
fig.tight_layout()

# Uranus
u1, v1 = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u1)*np.sin(v1)
y = np.sin(u1)*np.sin(v1)
z = np.cos(v1)
ax.plot_wireframe(x, y, z, color="r")


# Adding initial velocity and field vector, mostly for debugging
b_vec = b_initial/B0
ax.quiver(R[0, 0]/Ru, R[0, 1]/Ru, R[0, 2]/Ru, V[0, 0]/v0, V[0, 1]/v0, V[0, 2]/v0, color="r")
ax.quiver(R[0, 0]/Ru, R[0, 1]/Ru, R[0, 2]/Ru, b_vec[0], b_vec[1], b_vec[2], color="green")


# radial distance plot
f2, ax2 = plt.subplots()
rDist = np.linalg.norm(R, axis=1)
t = np.linspace(0, len(R), len(R))*dt
ax2.scatter(t, rDist/Ru, c="k", s=1)
ax2.set_xlabel("t [s]")
ax2.set_ylabel("Radial distance [Ru]")


# saving particle trace and velocity if desired
if savefile:
    if is_dipole:
        text = "dipole"
    else:
        text = "quadpole"
    fname = "../data/{}_{}_{}MeV_L{}_{}sec.txt".format(text, particleType, mev, L, seconds)
    j = 0
    while os.path.isfile(fname):
        fnameList = list(fname)
        if j == 0:
            fnameList[-5] = fnameList[-5] + "{}".format(j)
        else:
            fnameList[-5] = str(j)
        j += 1
        fname = "".join(fnameList)
    np.savetxt(fname, R)

if save_vel:
    if is_dipole:
        text = "dipole"
    else:
        text = "quadpole"
    fname = "../data/{}_{}_VELOCITY_{}MeV_L{}_{}sec.txt".format(text, particleType, mev, L, seconds)
    j = 0
    while os.path.isfile(fname):
        fnameList = list(fname)
        if j == 0:
            fnameList[-5] = fnameList[-5] + "{}".format(j)
        else:
            fnameList[-5] = str(j)
        j += 1
        fname = "".join(fnameList)
    np.savetxt(fname, V)


## JUST TO SEE INITIAL MAGNETIC FIELD AS A SLICE - ONLY FOR DEBUGGING
#xi = Ru*np.linspace(-6, 6, 10)
#zi = xi
#XX, YY, ZZ= np.meshgrid(xi, 0, zi)
#x = XX.ravel()
#y = YY.ravel()
#z = ZZ.ravel()
#RR = np.zeros([len(x), 3])
#RR[:, 0] = x
#RR[:, 1] = y
#RR[:, 2] = z
#
#fields = np.zeros(np.shape(RR))
#for i in range(len(fields)):
#    r = RR[i]
#    r_rotated = np.matmul(rotmat1, r)
#    B_rotated = Bquad(r_rotated, Ru, dipole=True)
#    B = np.matmul(rotmat2, B_rotated)
#    fields[i] = B
#
#ax.quiver(RR[:, 0]/Ru, RR[:, 1]/Ru, RR[:, 2]/Ru, fields[:, 0], np.zeros(len(fields)), fields[:, 2], normalize=True)
