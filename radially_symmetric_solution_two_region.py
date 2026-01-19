# Python Code For Radially Symmetric Numerical Solution for Buckling
# Organoid for two regions (cortex and subcortex)

# Kevin Roberts
# January 2026

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

#################################################
# Defining the material parameters and geometry #
#################################################
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
g_theta_c = 3.1 # growth multiplier in the circumferential direction in cortex region (#)
g_z = 1 # growth multiplier in the axial direction (#)

C_z = 1 # principal stretch in the axial direction (#)
P_f = 19.62 # external pressure (hydrostatic pressure) (Pa)
P_in = 0 # pressure inward normal to R_v boundary (Pa)
P_out = 0 # pressure inward normal to the R_c boundary (Pa)

# putting values for cortex and subcortex in a dictionary, used for different regions
subcortex_vals = dict(lam=lambda_s, mu=mu_s, g_r=g_r_s, g_theta=g_theta_s)
cortex_vals = dict(lam=lambda_c, mu=mu_c, g_r=g_r_c, g_theta=g_theta_c)

# For deciding where or not the inner subcortex boundary is fixed:
# "fixed" -> fixed boundary
# "pressure" -> pressure P_f applied normal to boundary
INNER_BC = "fixed"


#############################
# Defining stresses and eta #
#############################
def stresses(R, r, r_prime, params):

    # getting the parameters
    lam = params['lam']
    mu = params['mu']
    g_r = params['g_r']
    g_theta = params['g_theta']

    # avoiding division problems
    R = np.maximum(R, 1e-30)
    r = np.maximum(r, 1e-30)
    r_prime = np.maximum(r_prime, 1e-30)

    # computing the ln term and using it in the eta term
    ln = np.log((r_prime*(r/R)*C_z) / (g_r*g_theta*g_z))
    eta = lam * ln - mu

    P_RR = mu*(r_prime/(g_r**2)) + eta/r_prime
    P_ThetaTheta = mu*((r/R)/(g_theta**2)) + eta*(R/r)

    return P_RR, P_ThetaTheta, eta

################################
# Computing the partial derivs #
################################
# NOTE: ∂P_RR/∂R = ∂P_RR/∂R + (∂P_RR/∂r)*y0' + (∂P_RR/∂r')*y1', where y0 = r and y1 = r'
def partial_derivs(R, r, r_prime, eta, params):

    # getting the params
    lam = params['lam']
    mu = params['mu']
    g_r = params['g_r']

    # avoiding division problems
    R = np.maximum(R, 1e-30)
    r = np.maximum(r, 1e-30)
    r_prime = np.maximum(r_prime, 1e-30)

    # ∂P_RR/∂r'
    dP_dr_prime = (mu/(g_r**2)) + (lam-eta)/(r_prime**2)

    # ∂P_RR/∂r
    dP_dr = lam/(r_prime*r)

    # ∂P_RR/∂R
    dP_dR = -lam/(r_prime*R)

    return dP_dR, dP_dr, dP_dr_prime

#############################################################
# Defining the ODE function object for the solve_bvp solver #
#############################################################
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

        P_RR, P_ThetaTheta, eta = stresses(R, r, r_prime, params)
        dP_dR, dP_dr, dP_dr_prime = partial_derivs(R, r, r_prime, eta, params)

        # dP_RR/dR + (P_RR - P_ThetaTheta)/R = 0 ->
        # ∂P/∂R + (∂P/∂r)*r' + (∂P/∂r')*r'' + (P_RR - P_TT)/R = 0 ->
        # (∂P/∂r')*r'' = -(P_RR - P_TT)/R - ∂P/∂R - (∂P/∂r)*r'
        rhs = -(P_RR - P_ThetaTheta)/R - dP_dR - dP_dr*r_prime

        # Solve for r''
        r_double_prime = rhs/dP_dr_prime

        return np.vstack((r_prime, r_double_prime))
    return ode_system

# We need BVP solvers for each region given an interface displacement.
# We define the interface displacement as r_s

##########################################
# BVP solver for the subcortex given r_s #
##########################################
def solve_subcortex(r_s):
    ode = make_ode_system(subcortex_vals)

    R_mesh = np.linspace(R_v, R_s, 200)
    y_guess = np.vstack((np.linspace(R_v, r_s, R_mesh.size), np.ones_like(R_mesh)))

    # defining the boundary conditions for subcortex
    def bc(ya, yb):
        r_a = ya[0]
        rp_a = ya[1]
        r_b = yb[0]
        # rp_b = yb[1] # not used

        # boundary condition at the interface
        bc_int = r_b - r_s

        if INNER_BC == "fixed":
            bc_inner = r_a - R_v

        elif INNER_BC == "pressure":
            P_RR_a, _, _ = stresses(np.array([R_v]), np.array([r_a]), np.array([rp_a]), subcortex_vals)
            P_RR_a = P_RR_a[0]

            bc_inner = P_RR_a - P_in*C_z*(r_a/R_v)

        else:
            raise ValueError("INNER_BC must be 'fixed' or 'pressure'.")

        return np.array([bc_inner, bc_int])

    sol = solve_bvp(ode, bc, R_mesh, y_guess, max_nodes=5000, tol=1e-6)
    return sol

##########################################
# BVP solver for the subcortex given r_s #
##########################################
def solve_cortex(r_s):
    ode = make_ode_system(cortex_vals)

    R_mesh = np.linspace(R_s, R_c, 200)
    y_guess = np.vstack((np.linspace(r_s, R_c, R_mesh.size), np.ones_like(R_mesh)))

    def bc(ya, yb):
        r_a = ya[0]
        # rp_a = ya[1] # not used
        r_b = yb[0]
        rp_b = yb[1]

        # Boundary condition at the interface
        bc_int = r_a - r_s

        P_RR_b, _, _ = stresses(np.array([R_c]), np.array([r_b]), np.array([rp_b]), cortex_vals)
        P_RR_b = P_RR_b[0]

        bc_outer = P_RR_b + P_out*C_z*(r_b/R_c)

        return np.array([bc_int, bc_outer])

    sol = solve_bvp(ode, bc, R_mesh, y_guess, max_nodes=5000, tol=1e-6)
    return sol

########################################################################
# Stress jump function for root finding: P_RR^-(R_s) - P_RR^+(R_s) = 0 #
########################################################################
def stress_jump(r_s):
    sol1 = solve_subcortex(r_s)
    sol2 = solve_cortex(r_s)

    # if one of bvp didn't succeed, then return a big number to avoid bad guesses
    if (not sol1.success) or (not sol2.success):
        print("One or both of the bvp did not succeed.")
        return np.sign(r_s - R_s)*1e9

    # Evaluate each side at the interface
    r1_s, rp1_s = sol1.sol(np.array([R_s]))
    r2_s, rp2_s = sol2.sol(np.array([R_s]))

    P_RR_1, _, _ = stresses(np.array([R_s]), np.array([r1_s[0]]), np.array([rp1_s[0]]), subcortex_vals)
    P_RR_2, _, _ = stresses(np.array([R_s]), np.array([r2_s[0]]), np.array([rp2_s[0]]), cortex_vals)

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
cortex_solution = solve_cortex(r_s_star)
subcortex_solution = solve_subcortex(r_s_star)

print("Subcortex converged:", cortex_solution.success, cortex_solution.message)
print("Cortex converged:   ", subcortex_solution.success, subcortex_solution.message)

# -----------------------------
# Stitch solutions for plotting
# -----------------------------
R1 = np.linspace(R_v, R_s, 300)
R2 = np.linspace(R_s, R_c, 300)

# These are 1D arrays already (length 300)
r1, rp1 = cortex_solution.sol(R1)
r2, rp2 = subcortex_solution.sol(R2)

# Stitch (skip the duplicate interface point in region 2)
R_all  = np.concatenate([R1, R2[1:]])
r_all  = np.concatenate([r1, r2[1:]])
rp_all = np.concatenate([rp1, rp2[1:]])

# Stresses (pass 1D arrays)
P_RR_1, P_TT_1, _ = stresses(R1, r1, rp1, subcortex_vals)
P_RR_2, P_TT_2, _ = stresses(R2, r2, rp2, cortex_vals)

P_RR_all = np.concatenate([P_RR_1, P_RR_2[1:]])
P_TT_all = np.concatenate([P_TT_1, P_TT_2[1:]])


#######################################################
# Checking for Instability: if lam_r/lam_theta >= 2.4 #
#######################################################
def instability_check(cortex_solution, subcortex_solution):
    # check at the outer surface R = R_c
    y = cortex_solution.sol(np.array([R_c]))
    r_out = float(y[0][0])
    rp_out = float(y[1][0])

    # computing the total stretches
    F_RR = rp_out
    F_ThetaTheta = r_out/R_c

    # elastic stretches
    g_r = cortex_vals['g_r']
    g_theta = cortex_vals['g_theta']

    # lam_r and lam_theta are the ELASTIC principle stretches
    lam_r = F_RR / g_r
    lam_theta = F_ThetaTheta / g_theta

    ratio = lam_r / lam_theta

    print(f"Instability ratio: {ratio}")

instability_check(cortex_solution, subcortex_solution)



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
r_in  = cortex_solution.sol(np.array([R_v]))[0][0]
r_s   = r_s_star
r_out = subcortex_solution.sol(np.array([R_c]))[0][0]

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
