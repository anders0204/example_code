# Example Code
### Anders Varmann Aamodt

## Uranus_quadpole_sims_v11
This is the main code used to trace particles in the Uranian magnetosphere.
The program solves the equation of motion for the charged particles using a sixth-order Runge-Kutta method.
The magnetic field of Uranus is implemented with a spherical harmonics model, which allows the possibility of
either expanding to the dipole component, or the full dipole+quadrupole field which represents the real Uranian magnetic field.

The result is a trace of the particle around the planet, in this case only the dipole component is used:
![](https://www.linkpicture.com/q/uranus_dipole_trace.png)

## quad_vs_dip
This is the code used to investigate where, and how significant the differences are between a pure dipole field
and the full Uranian (dipole+quadrupole) magnetic field. The program creates a grid of test points in spherical
coordinates around the planet, and calculates both the magnitude (field strength) difference, and the angular (vector) difference
between the fields. These differences are then mapped to 2D grids in 6 "phi-slices" and plotted.

Example result:
![](https://www.linkpicture.com/q/magnitude_difference.png)

