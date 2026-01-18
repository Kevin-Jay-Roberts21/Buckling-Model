# Python Code For Radially Symmetric Numerical Solution for Buckling
# Organoid for only one (cortex) region

# Kevin Roberts
# January 2026

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Defining the material parameters
R_v = 50 # void radius (um)
R_c = 200 # cortex radius (um)

lambd = 3000 # bulk modulus (Pa)
mu = 34200 # shear modulus (Pa)

g_r = 1 # growth multiplier in the radial direction (#)
g_theta = 1.2 # growth multiplier in the circumferential direction (#)
g_z = 1 # growth multiplier in the axial direction (#)

C_z = 1 # principal stretch in the axial direction (#)
P_f = 19.62 # external pressure (hydrostatic pressure) (Pa)
P_in = 0 # pressure inward normal to R_v boundary (Pa)
P_out = 0 # pressure inward normal to the R_c boundary (Pa)


# Computing the stresses and natural log terms in terms of R, r, and r_prime
def stresses(R, r, r_prime):

    # avoiding division problems
    R = np.maximum(R, 1e-30)
    r = np.maximum(r, 1e-30)
    r_prime = np.maximum(r_prime, 1e-30)

    # computing the ln term and using it in the eta term
    ln = np.log((r_prime*(r/R)*C_z) / (g_r*g_theta*g_z))
    eta = lambd * ln - mu

    P_RR = mu*(r_prime/(g_r**2)) + eta/r_prime
    P_ThetaTheta = mu*((r/R)/(g_theta**2)) + eta*(R/r)

    return P_RR, P_ThetaTheta, ln, eta

# Computing the partial derivatives in the equilibrium radial equation. They are needed from the chain rule since:
# ∂P_RR/∂R = ∂P_RR/∂R + (∂P_RR/∂r)*y0' + (∂P_RR/∂r')*y1', where y0 = r and y1 = r'
def partial_derivs(R, r, r_prime, ln, eta):

    # avoiding division problems
    R = np.maximum(R, 1e-30)
    r = np.maximum(r, 1e-30)
    r_prime = np.maximum(r_prime, 1e-30)

    # ∂P_RR/∂r'
    dP_dr_prime = (mu/(g_r**2)) + (lambd-eta)/(r_prime**2)

    # ∂P_RR/∂r
    dP_dr = lambd/(r_prime*r)

    # ∂P_RR/∂R
    dP_dR = -lambd/(r_prime*R)

    return dP_dR, dP_dr, dP_dr_prime

# Defining the ODE system for the solve_bvp solver
def ode_system(R, y):

    # The following system to be solved is defined as follows:
    # y[0] = r(R)
    # y[1] = r_prime(R)
    # and we will return y' = [r', r'']

    # avoiding division issues
    R = np.maximum(R, 1e-30)

    r = y[0]
    r_prime = y[1]

    P_RR, P_ThetaTheta, ln, eta = stresses(R, r, r_prime)
    dP_dR, dP_dr, dP_dr_prime = partial_derivs(R, r, r_prime, ln, eta)

    # dP_RR/dR + (P_RR - P_ThetaTheta)/R = 0 -->
    # ∂P/∂R + (∂P/∂r)*r' + (∂P/∂r')*r'' + (P_RR - P_TT)/R = 0 -->
    # (∂P/∂r')*r'' = -(P_RR - P_TT)/R - ∂P/∂R - (∂P/∂r)*r'
    rhs = -(P_RR - P_ThetaTheta)/R - dP_dR - dP_dr*r_prime

    # Solve for r''
    r_double_prime = rhs/dP_dr_prime

    return np.vstack((r_prime, r_double_prime))


# fixed inner boundary condition, pressure on outer cortex boundary
def boundary_conditions_inner_fixed(ya, yb):
    # NOTE: ya = r(R_v) and yb = r(Rc)

    # BC1: r(R_v) = R_v (fixed)
    # BC2: PRR(R_c) = -P_f * C_z * r(R_c)/R_c (pressure on outer cortex boundary)

    r_a, r_prime_a = ya
    r_b, r_prime_b = yb

    P_RR_b, _, _, _ = stresses(np.array([R_c]), np.array([r_b]), np.array([r_prime_b]))
    P_RR_b = P_RR_b[0]

    bc1 = r_a - R_v
    bc2 = P_RR_b + P_f*C_z*(r_b/R_c)

    return np.array([bc1, bc2])

# pressure on outer cortex boundary and inner cortex boundary
def boundary_conditions_neither_fixed(ya, yb):
    # NOTE: ya = r(R_v) and yb = r(Rc)

    # BC1: PRR(R_v) = P_f * C_z * r(R_v)/R_v
    # BC2: PRR(R_c) = -P_f * C_z * r(R_c)/R_c (pressure on outer cortex boundary)

    r_a, r_prime_a = ya
    r_b, r_prime_b = yb

    P_RR_a, _, _, _ = stresses(np.array([R_v]), np.array([r_a]), np.array([r_prime_a]))
    P_RR_b, _, _, _ = stresses(np.array([R_c]), np.array([r_b]), np.array([r_prime_b]))
    P_RR_a = P_RR_a[0]
    P_RR_b = P_RR_b[0]

    bc1 = P_RR_a - P_in*C_z*(r_a/R_v)
    bc2 = P_RR_b + P_out*C_z*(r_b/R_c)

    return np.array([bc1, bc2])

# Initial Mesh and initial guess
################################

# mesh
R_mesh = np.linspace(R_v, R_c, 200)

# simple initial guess
y_guess = np.vstack((R_mesh.copy(), np.ones_like(R_mesh)))

# Solving
sol = solve_bvp(ode_system, boundary_conditions_inner_fixed, R_mesh, y_guess, max_nodes=5000, tol=1e-6)

print("Converged: ", sol.success)
print("Message: ", sol.message)



# Plotting
def plot_reference_vs_deformed_boundaries(sol, Rv, Rc, title="Reference vs deformed boundaries"):

    theta = np.linspace(0, 2*np.pi, 600)

    # Reference boundaries
    x_in_ref = Rv * np.cos(theta); y_in_ref = Rv * np.sin(theta)
    x_out_ref = Rc * np.cos(theta); y_out_ref = Rc * np.sin(theta)

    # Deformed boundaries (evaluate r at Rv and Rc)
    r_in = sol.sol(np.array([Rv]))[0][0]
    r_out = sol.sol(np.array([Rc]))[0][0]

    x_in_def = r_in * np.cos(theta); y_in_def = r_in * np.sin(theta)
    x_out_def = r_out * np.cos(theta); y_out_def = r_out * np.sin(theta)

    plt.figure()
    plt.plot(x_in_ref, y_in_ref, "--", color="blue", label="Inner ref (R=Rv)")
    plt.plot(x_out_ref, y_out_ref, "--", color="red", label="Outer ref (R=Rc)")
    plt.plot(x_in_def, y_in_def, "-", color="blue", label="Inner deformed (r(Rv))")
    plt.plot(x_out_def, y_out_def, "-", color="red", label="Outer deformed (r(Rc))")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_reference_vs_deformed_boundaries(sol, R_v, R_c, title="Reference vs deformed boundaries")


R_plot = np.linspace(R_v, R_c, 400)
r_plot, r_prime_plot = sol.sol(R_plot)

P_RR_plot, P_ThetaTheta_plot, _, _ = stresses(R_plot, r_plot, r_prime_plot)

plt.figure()
plt.plot(R_plot, r_plot)
plt.xlabel("R")
plt.ylabel("r(R)")
plt.title("Deformed radius mapping")
plt.grid(True)

plt.figure()
plt.plot(R_plot, P_RR_plot, label="P_RR")
plt.plot(R_plot, P_ThetaTheta_plot, label="P_ThetaTheta")
plt.xlabel("R")
plt.ylabel("Piola stress")
plt.title("Stress components")
plt.legend()
plt.grid(True)

u_plot = r_plot - R_plot

plt.figure()
plt.plot(R_plot, u_plot, linewidth=2)
plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.xlabel("R")
plt.ylabel("u(R) = r(R) - R")
plt.title("Radial displacement")
plt.grid(True)
plt.show()

plt.show()