import numpy as np
import matplotlib.pyplot as plt
import time

t0 = time.time()
plt.close("all")
#plt.style.use(["science", "no-latex"])

def nearest(coords, dat):
    """ Function to match a coordinate to the closest coordinate in a dataset.
    
    Input:
        coords (tuple): Coordinate (x, y) to match.
        dat (arr): 2xN array with a list of coordinate points.
    
    Output:
        ind (int): Index of the row in dat which is the closest to coords.
    """
    x, y = coords
    X = dat[:, 0]/Ru
    Y = dat[:, 1]/Ru
    ind = np.abs(np.sqrt((X-x)**2+(Y-y)**2)).argmin()
    return ind


def rotmat(axis, ang):
    """ Function to generate a rotation matrix about the three principal axis
    """
    if axis == "x":
        return np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
    if axis == "y":
        return np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    if axis == "z":
        return np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    

def norm(vec, axis=1):
    return np.linalg.norm(vec, axis=axis)


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

# Specifiying the Uranian equatorial radius in metres
Ru = 25362e3

# number of Ru to look at
n = 3
min_r = 1.1 # minimum radius in Ru

# generating a grid of test points in spherical coordinates
radii = Ru*np.arange(min_r, n, 0.1)
phis = np.linspace(0, np.pi-np.deg2rad(30), 6)
n_thetas = 80
min_theta = 5 # avoiding singularity at +- pi/2
thetas = np.concatenate((np.linspace(0+np.deg2rad(min_theta), np.pi-(np.deg2rad(min_theta)), 
                                     n_thetas), np.linspace(np.pi+np.deg2rad(min_theta), 2*np.pi-(np.deg2rad(min_theta)), n_thetas)))
rad, th, ph = np.meshgrid(radii, thetas, phis)
pos = np.vstack([rad.ravel(), th.ravel(), ph.ravel()])
positions_ = pos.T
positions = np.zeros([len(positions_), 3])

# converting spherical coordinates to Cartesian
for i in range(len(positions_)):
    r, theta, phi = positions_[i]
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    positions[i] = [x, y, z]

N = len(positions)

# separating slices into dictionary, collecting corrdinates of the same slices
slices = dict()
ii = 0
for pos in positions_:
    phi = pos[2]
    if phi not in slices.keys():
        slices[phi] = list()
        slices[phi].append(positions[ii])
    else:
        slices[phi].append(positions[ii])
    ii += 1

for key in slices.keys():
    slices[key] = np.array(slices[key])



# creating dictionaries of field differences for each slice
slices_mag_diff = dict()
slices_vec_diff = dict()
for phi in phis:
    pos = slices[phi]
    NN = len(pos)
    slices_mag_diff[phi] = np.zeros(NN)
    slices_vec_diff[phi] = np.zeros(NN)
    
    for i in range(NN):
        real_pos = pos[i]
        rotated_pos = np.matmul(rotmat1, real_pos)
        rotated_dip = Bquad(rotated_pos, Ru, dipole=True)
        rotated_quad = Bquad(rotated_pos, Ru, dipole=False)
        dipole = np.matmul(rotmat2, rotated_dip)
        quadrupole = np.matmul(rotmat2, rotated_quad)
        
        # this time calculating percentage quadpole difference to dipole
        slices_mag_diff[phi][i] = np.linalg.norm(quadrupole)/np.linalg.norm(dipole)
        
        # vector difference is just angle
        dot = np.dot(dipole, quadrupole)
        slices_vec_diff[phi][i] = np.rad2deg(np.arccos(dot/(np.linalg.norm(dipole)*np.linalg.norm(quadrupole))))


# creating a dictionary of (r, z) coordinates for each slice, where r is the
# horizontal axis (negative to the left, positive to the right), and matching
# these coordinates with two other dictionaries with corresponding difference
# values
data = dict() # this will contain the coordinate points
slices_mag = dict()
slices_vec = dict()
for s in range(len(phis)):
    coor = slices[phis[s]]
    data[s] = np.zeros([NN, 2])
    for i in range(NN):
        x, y, z = coor[i]
        if y > 0:
            fac = 1
        elif y < 0:
            fac = -1
        if y == 0:
            if x > 0:
                fac = 1
            else:
                fac = -1
        r = fac*np.sqrt(x**2+y**2)
        data[s][i] = [r, -z]
    
    # final dictionaries containing the field differences
    slices_mag[s] = slices_mag_diff[phis[s]]
    slices_vec[s] = slices_vec_diff[phis[s]]
    

# plotting
f, axs = plt.subplots(2, 3)
f2, axs2 = plt.subplots(2, 3)
label_vec = r"$B_{quad}$" + " vs. " + r"$B_{dip}$" + " angular difference" + "  ["+ r"$^\circ$" + "]" 
label_mag = r"$\vert B_{quad}\vert / \vert B_{dip} \vert$" 
f.suptitle(label_vec, fontsize=18)
f2.suptitle(label_mag, fontsize=18)

# creating grid of pixels
grid_dim = 100 # resolution
xi = np.linspace(-n,n,grid_dim)
yi = np.linspace(-n,n,grid_dim)

nSlices = len(data)
grid_vec = np.zeros([nSlices, grid_dim, grid_dim])
grid_mag = np.zeros([nSlices, grid_dim, grid_dim])

# indices to select the correct axes
ii = 0
jj = 0
for s in range(nSlices):
    # matching x-y grid points to closest coordinate with a measured field  
    for i in range(grid_dim):
        for j in range(grid_dim):
            x = xi[i]
            y = yi[j]
            if np.sqrt(x**2+y**2) < min_r:
                grid_vec[s, i, j] = None
                grid_mag[s, i, j] = None

            else:
                ind = nearest((x, y), data[s])
                grid_vec[s, i, j] = slices_vec[s][ind]
                grid_mag[s, i, j] = slices_mag[s][ind]
    
    # plotting vector difference        
    axs[ii, jj].set_aspect("equal")
    title = r"$\phi \;= $" + " {:.1f}".format(np.rad2deg(phis[s])) + r"$^\circ$" + "," + " {:.1f}".format(np.rad2deg(phis[s]+np.pi)) + r"$^\circ$"
    axs[ii, jj].set_title(title, fontsize=14)
    planet = plt.Circle((0,0), 1, color="lightcyan", ec="k") # creating planet
    axs[ii, jj].add_patch(planet)
    
    im = axs[ii, jj].imshow(grid_vec[s].T, extent=[-n, n, -n, n], interpolation="none")
    f.colorbar(im, ax=axs[ii, jj])   
    
    if s > 2:
        axs[ii, jj].set_xlabel("r [Ru]", fontsize=18)
    if s == 0 or s == 3:
        axs[ii, jj].set_ylabel("z [Ru]", fontsize=18)
    
    # plotting mag difference
    axs2[ii, jj].set_aspect("equal")
    axs2[ii, jj].set_title(title)
    planet = plt.Circle((0,0), 1, color="lightcyan", ec="k") # creating planet
    axs2[ii, jj].add_patch(planet)
    
    im = axs2[ii, jj].imshow(grid_mag[s].T, extent=[-n, n, -n, n], interpolation="none")
    f2.colorbar(im, ax=axs2[ii, jj])   
    
    if s > 2:
        axs2[ii, jj].set_xlabel("r [Ru]", fontsize=14)
    if s == 0 or s == 3:
        axs2[ii, jj].set_ylabel("z [Ru]", fontsize=14)   
    
    if jj == 2:
        jj = 0
        ii = 1
    else:    
        jj += 1


t = time.time()
print("That took {:.1f} s.".format(t-t0))
