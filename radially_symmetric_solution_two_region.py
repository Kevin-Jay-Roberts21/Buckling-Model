# Python Code For Radially Symmetric Numerical Solution for Buckling
# Organoid for two regions (cortex and subcortex)

# Kevin Roberts
# January 2026

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Defining the material parameters
R_v = 50 # void radius (um)
R_s = 150 # subcortex radius (um)
R_c = 200 # cortex radius (um)

lambda_s = 1100 # bulk modulus in subcortex region (Pa)
lambda_c = 3000 # bulk modulus in cortex region (Pa)
mu_s = 11400 # shear modulus in subcortex region (Pa)
mu_c = 34200 # shear modulus in cortex region (Pa)

g_r_s = 1.05 # growth multiplier in the radial direction in subcortex region (#)
g_r_c = 1.1 # growth multiplier in the radial direction in cortex region (#)
g_theta_s = 1.05 # growth multiplier in the circumferential direction in subcortex region (#)
g_theta_c = 1.5 # growth multiplier in the circumferential direction in cortex region (#)
g_z = 1 # growth multiplier in the axial direction (#)

C_z = 1 # principal stretch in the axial direction (#)
P_f = 19.62 # external pressure (hydrostatic pressure) (Pa)
P_in = 0 # pressure inward normal to R_v boundary (Pa)
P_out = 0 # pressure inward normal to the R_c boundary (Pa)

# putting values for cortex and subcortex in a dictionary
subcortex_vals = dict(lambd=lambda_s, mu=mu_s, g_r=g_r_s, g_theta=g_theta_s, g_z=g_z)
cortex_vals = dict(lambd=lambda_c, mu=mu_c, g_r=g_r_c, g_theta=g_theta_c, g_z=g_z)

# For deciding where or not the inner boundary is fixed:
# "fixed" for fixed boundary
# "pressure" for pressure P_f applied to boundary
INNER_BC = "fixed"

# Computing the stresses and natural log terms in terms of R, r, and r_prime
def stresses(R, r, r_prime, params):

    # getting the parameters
    lambd = params['lambd']
    mu = params['mu']
    g_r = params['g_r']
    g_theta = params['g_theta']
    g_z = params['g_z']

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
def partial_derivs(R, r, r_prime, ln, eta, params):

    # getting the params
    lambd = params['lambd']
    mu = params['mu']
    g_r = params['g_r']

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

# Defining the ODE system for the solve_bvp solver and defining a "make" ode_system
# this has to be done inside the function since params is piecewise. We don't need
# this design in the one region case

def make_ode_system(params):
    def ode_system(R, y):

        # The following system to be solved is defined as follows:
        # y[0] = r(R)
        # y[1] = r_prime(R)
        # and we will return y' = [r', r'']

        # avoiding division issues
        R = np.maximum(R, 1e-30)

        r = y[0]
        r_prime = y[1]

        P_RR, P_ThetaTheta, ln, eta = stresses(R, r, r_prime, params)
        dP_dR, dP_dr, dP_dr_prime = partial_derivs(R, r, r_prime, ln, eta, params)

        # dP_RR/dR + (P_RR - P_ThetaTheta)/R = 0 -->
        # ∂P/∂R + (∂P/∂r)*r' + (∂P/∂r')*r'' + (P_RR - P_TT)/R = 0 -->
        # (∂P/∂r')*r'' = -(P_RR - P_TT)/R - ∂P/∂R - (∂P/∂r)*r'
        rhs = -(P_RR - P_ThetaTheta)/R - dP_dR - dP_dr*r_prime

        # Solve for r''
        r_double_prime = rhs/dP_dr_prime

        return np.vstack((r_prime, r_double_prime))
    return ode_system

# We need BVP solvers for each region given an interface displacement, which is defined as r_s

# BVP solver for the subcortex given r_s
def solve_subcortex(r_s):
    ode = make_ode_system(subcortex_vals)

    R_mesh = np.linspace(R_v, R_s, 200)
    y_guess = np.vstack((np.linspace(R_v, r_s, R_mesh.size), np.ones_like(R_mesh)))

    # similar to bc function in "one region" code with modification for fixed and not fixed inner boundary
    def bc(ya, yb):
        r_a, rp_a = ya
        r_b, rp_b = yb

        # boundary condition at the interface
        bc_int = r_b - r_s

        if INNER_BC == "fixed":
            bc_inner = r_a - R_v

        elif INNER_BC == "pressure":
            P_RR_a, _, _, _ = stresses(np.array([R_v]), np.array([r_a]), np.array([rp_a]), subcortex_vals)
            P_RR_a = P_RR_a[0]

            bc_inner = P_RR_a - P_in*C_z*(r_a/R_v)

        else:
            raise ValueError("INNER_BC must be 'fixed' or 'pressure'.")

        return np.array([bc_inner, bc_int])

    sol = solve_bvp(ode, bc, R_mesh, y_guess, max_nodes=5000, tol=1e-6)
    return sol

# BVP solver for the subcortex given r_s
def solve_cortex(r_s):
    ode = make_ode_system(cortex_vals)

    R_mesh = np.linspace(R_s, R_c, 200)
    y_guess = np.vstack((np.linspace(r_s, R_c, R_mesh.size), np.ones_like(R_mesh)))

    def bc(ya, yb):
        r_a, rp_a = ya
        r_b, rp_b = yb

        # Boundary condition at the interface
        bc_int = r_a - r_s

        P_RR_b, _, _, _ = stresses(np.array([R_c]), np.array([r_b]), np.array([rp_b]), cortex_vals)
        P_RR_b = P_RR_b[0]

        bc_outer = P_RR_b + P_out*C_z*(r_b/R_c)

        return np.array([bc_int, bc_outer])

    sol = solve_bvp(ode, bc, R_mesh, y_guess, max_nodes=5000, tol=1e-6)
    return sol

# Adding a stress jump function for rooting finding of P_RR^-(R_s) - P_RR^+(R_s)
def stress_jump(r_s):
    sol1 = solve_subcortex(r_s)
    sol2 = solve_cortex(r_s)

    # is neither bvp succeeded, then return a big number to avoid bad guesses
    if (not sol1.success) or (not sol2.success):
        return np.sign(r_s - R_s)*1e9

    # Evaluate each side at the interface
    r1_s, rp1_s = sol1.sol(np.array([R_s]))
    r2_s, rp2_s = sol2.sol(np.array([R_s]))

    P_RR_1, _, _, _ = stresses(np.array([R_s]), np.array([r1_s[0]]), np.array([rp1_s[0]]), subcortex_vals)
    P_RR_2, _, _, _ = stresses(np.array([R_s]), np.array([r2_s[0]]), np.array([rp2_s[0]]), cortex_vals)

    return P_RR_1[0] - P_RR_2[0]

# Now find r_s such that the traction is continuous
br_lo = max(1e-6, 0.2 * R_s)
br_hi = 2.0 * R_s

# Expand bracket until sign change (simple robust bracketing)
f_lo = stress_jump(br_lo)
f_hi = stress_jump(br_hi)

tries = 0
while f_lo * f_hi > 0 and tries < 12:
    br_lo *= 0.7
    br_hi *= 1.3
    f_lo = stress_jump(br_lo)
    f_hi = stress_jump(br_hi)
    tries += 1

if f_lo * f_hi > 0:
    raise RuntimeError(
        "Could not bracket a root for r_s. Try different initial ranges or milder parameters."
    )

root = root_scalar(stress_jump, bracket=(br_lo, br_hi), method="brentq", xtol=1e-6)
r_s_star = root.root
print("Interface displacement r(R_s) =", r_s_star)

# Solve one last time with the matched interface displacement
sol_sub = solve_subcortex(r_s_star)
sol_ctx = solve_cortex(r_s_star)

print("Subcortex converged:", sol_sub.success, sol_sub.message)
print("Cortex converged:   ", sol_ctx.success, sol_ctx.message)

# -----------------------------
# Stitch solutions for plotting
# -----------------------------
R1 = np.linspace(R_v, R_s, 300)
R2 = np.linspace(R_s, R_c, 300)

# These are 1D arrays already (length 300)
r1, rp1 = sol_sub.sol(R1)
r2, rp2 = sol_ctx.sol(R2)

# Stitch (skip the duplicate interface point in region 2)
R_all  = np.concatenate([R1, R2[1:]])
r_all  = np.concatenate([r1, r2[1:]])
rp_all = np.concatenate([rp1, rp2[1:]])

# Stresses (pass 1D arrays)
P_RR_1, P_TT_1, _, _ = stresses(R1, r1, rp1, subcortex_vals)
P_RR_2, P_TT_2, _, _ = stresses(R2, r2, rp2, cortex_vals)

P_RR_all = np.concatenate([P_RR_1, P_RR_2[1:]])
P_TT_all = np.concatenate([P_TT_1, P_TT_2[1:]])



# PLOTTING

# -----------------------------
# Plots (same style as your 1-region)
# -----------------------------
def plot_reference_vs_deformed_boundaries_two_region(
    r_in, r_s, r_out,
    title="Reference vs deformed boundaries (two-region)"
):
    theta = np.linspace(0, 2*np.pi, 600)

    # --- Reference circles ---
    x_in_ref  = R_v * np.cos(theta); y_in_ref  = R_v * np.sin(theta)
    x_s_ref   = R_s * np.cos(theta); y_s_ref   = R_s * np.sin(theta)
    x_out_ref = R_c * np.cos(theta); y_out_ref = R_c * np.sin(theta)

    # --- Deformed circles ---
    x_in_def  = r_in  * np.cos(theta); y_in_def  = r_in  * np.sin(theta)
    x_s_def   = r_s   * np.cos(theta); y_s_def   = r_s   * np.sin(theta)
    x_out_def = r_out * np.cos(theta); y_out_def = r_out * np.sin(theta)

    plt.figure(figsize=(6, 6))

    # Reference (dashed)
    plt.plot(x_in_ref,  y_in_ref,  "--", color="blue",  alpha=0.5, label="Inner ref (R=Rv)")
    plt.plot(x_s_ref,   y_s_ref,   "--", color="green", alpha=0.5, label="Interface ref (R=Rs)")
    plt.plot(x_out_ref, y_out_ref, "--", color="red",   alpha=0.5, label="Outer ref (R=Rc)")

    # Deformed (solid)
    plt.plot(x_in_def,  y_in_def,  "-", color="blue",  linewidth=2, label="Inner deformed r(Rv)")
    plt.plot(x_s_def,   y_s_def,   "-", color="green", linewidth=2, label="Interface deformed r(Rs)")
    plt.plot(x_out_def, y_out_def, "-", color="red",   linewidth=2, label="Outer deformed r(Rc)")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# inner, interface and outer deformed radii
r_in  = sol_sub.sol(np.array([R_v]))[0][0]
r_s   = r_s_star
r_out = sol_ctx.sol(np.array([R_c]))[0][0]

plot_reference_vs_deformed_boundaries_two_region(r_in, r_s, r_out)

# r(R)
plt.figure()
plt.plot(R_all, r_all)
plt.axvline(R_s, linestyle="--", color="k", linewidth=1, label="Interface R_s")
plt.xlabel("R"); plt.ylabel("r(R)")
plt.title("Deformed radius mapping (two-region)")
plt.grid(True); plt.legend()
plt.show()

# stresses
plt.figure()
plt.plot(R_all, P_RR_all, label="P_RR")
plt.plot(R_all, P_TT_all, label="P_ThetaTheta")
plt.axvline(R_s, linestyle="--", color="k", linewidth=1, label="Interface R_s")
plt.xlabel("R"); plt.ylabel("Piola stress")
plt.title("Stress components (two-region)")
plt.grid(True); plt.legend()
plt.show()

# displacement
u_all = r_all - R_all
plt.figure()
plt.plot(R_all, u_all, linewidth=2)
plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.axvline(R_s, linestyle="--", color="k", linewidth=1, label="Interface R_s")
plt.xlabel("R"); plt.ylabel("u(R)=r(R)-R")
plt.title("Radial displacement (two-region)")
plt.grid(True); plt.legend()
plt.show()
