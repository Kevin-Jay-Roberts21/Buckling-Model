# Incremental stability analysis using a Stroh first-order system in R.
# Code uses radially_symmetric_solution_two_region.py

# Kevin Roberts
# January 2026

import numpy as np
from numpy.linalg import det, solve, inv
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

import radially_symmetric_solution_two_region as base

###########################################################
# USING RADIALLY SYMMETRIC SOLUTION TO GET THE BASE STATE #
###########################################################
# For convenience, we write a "base state" wrapper as a
# class, getting all of the information from the radially
# symmetric solution
class BaseState:
    # Holds the piecewise base solution r(R), r'(R) from solve_bvp from the radially_symmetric_two_region.py code
    def __init__(self, subcortex_sol, cortex_sol):
        self.subcortex_sol = subcortex_sol
        self.cortex_sol = cortex_sol

    def base_state_params_given_R(self, R):
        # returns (r, r_prime, region_name) at scalar R
        R = float(R)

        # if the given R is in the subcortex region
        if R <= base.R_s:
            r, r_prime = self.subcortex_sol.sol(np.array([R]))
            return float(r[0]), float(r_prime[0]), "subcortex"
        # if the given R is in the coretx region
        else:
            r, r_prime = self.cortex_sol.sol(np.array([R]))
            return float(r[0]), float(r_prime[0]), "cortex"

def solve_base_state_for_gthetac(g_theta_c):

    # Update cortex growth multiplier
    base.cortex_vals["g_theta"] = float(g_theta_c)

    # Solve the base state using the radial solver's function
    sub_sol, cor_sol, r_s_star = base.solve_base_state()

    return BaseState(subcortex_sol=sub_sol, cortex_sol=cor_sol)


######################################################
# defining linearized, Fourier mode of F: deltaF_hat #
######################################################
def deltaF_hat(R, uhat, uhat_prime, m, k):

    # getting R and defining imaginary numbers
    R = max(float(R), 1e-12)
    i = 1j

    # defining the uhat vector and uhatprime vector
    u_r, u_theta, u_z = uhat
    u_r_prime, u_theta_prime, u_z_prime = uhat_prime

    # defining the delta F hat matrix
    dF_hat = np.zeros((3,3), dtype=complex)

    # defining the components of the delta F hat matrix (first row)
    dF_hat[0, 0] = u_r_prime
    dF_hat[0, 1] = (i*m/R)*u_r - u_theta/R
    dF_hat[0, 2] = (i*k)*u_r

    # second row
    dF_hat[1, 0] = u_theta_prime
    dF_hat[1, 1] = (i*m/R)*u_theta + u_r/R
    dF_hat[1, 2] = (i*k)*u_theta

    # third row
    dF_hat[2, 0] = u_z_prime
    dF_hat[2, 1] = (i*m/R)*u_z
    dF_hat[2, 2] = (i*k)*u_z

    return dF_hat

####################################
# Defining linearized Piola stress #
####################################
def deltaP_hat(dF, F_0, Fg, lambd, mu):

    # Fg matrix computations
    Fg_inverse = inv(Fg)
    Fg_inverse_transpose = Fg_inverse.T

    # Fe_0 matrix definition and computations (the symbol @ means matrix multiplication)
    Fe_0 = F_0 @ Fg_inverse
    Fe_0_inverse = inv(Fe_0)
    Fe_0_inverse_transpose = Fe_0_inverse.T
    Je_0 = det(Fe_0)
    ln_Je_0 = np.log(Je_0)

    # there are three terms in deltaP_hat added together. We wil define them, add
    # them and then multiply the summation by Fg_inverse_transpose:

    term1 = mu*(dF @ Fg_inverse)
    term2 = lambd*np.trace(Fe_0_inverse @ dF @ Fg_inverse)*Fe_0_inverse_transpose
    term3 = -(lambd*ln_Je_0 - mu)*(Fe_0_inverse_transpose @ Fg_inverse_transpose @ dF.T @ Fe_0_inverse_transpose)

    dP_hat = (term1 + term2 + term3)@Fg_inverse_transpose

    return dP_hat

##########################################################
# Defining the rhs of the linearized boundary conditions #
##########################################################
def boundary_condition_rhs(dF, F_0, P_f, N_R=+1.0):

    # defining the F_0 matrix computations
    F_0_inverse = inv(F_0)
    F_0_inverse_transpose = F_0_inverse.T
    J_0 = det(F_0)

    # defining the normal vector N (Note: the Nsign is +1 if outer boundary, and -1 if inner boundary)
    N = np.array([N_R, 0, 0], dtype=complex)

    rhs = -P_f*J_0*((np.trace(F_0_inverse@dF)*F_0_inverse_transpose - (F_0_inverse_transpose @ dF.T @ F_0_inverse_transpose)) @ N)

    return rhs

###########################
# Stroh matrices: Q and W #
###########################
def stroh_QR_matrices(R, F_0, Fg, lambd, mu, m, k):

    # build W from: t = Wu when u' = 0
    W = np.zeros((3, 3), dtype=complex)
    uhat_prime_0 = np.zeros(3, dtype=complex)
    for j in range(3):
        ej = np.zeros(3, dtype=complex)
        ej[j] = 1.0
        dF = deltaF_hat(R, ej, uhat_prime_0, m, k)
        dP = deltaP_hat(dF, F_0, Fg, lambd, mu)
        W[:, j] = np.array([dP[0, 0], dP[1, 0], dP[2, 0]], dtype=complex)

    # build Q from: t = Qu' when u = 0
    Q = np.zeros((3, 3), dtype=complex)
    uhat_0 = np.zeros(3, dtype=complex)
    for j in range(3):
        ej = np.zeros(3, dtype=complex)
        ej[j] = 1.0
        dF = deltaF_hat(R, uhat_0, ej, m, k)
        dP = deltaP_hat(dF, F_0, Fg, lambd, mu)
        Q[:, j] = np.array([dP[0, 0], dP[1, 0], dP[2, 0]], dtype=complex)

    return Q, W

#######################################
# Calculating u' = Q^{-1}(that - W u) #
#######################################
def reconstruct_uhat_prime(R, uhat, that, F_0, Fg, lambd, mu, m, k):

    Q, W = stroh_QR_matrices(R, F_0, Fg, lambd, mu, m, k)
    uhat_prime = solve(Q, that - W @ uhat)

    return uhat_prime


####################################################
# Build the Stroh-style ODE: eta' = (uhat', that') #
####################################################
def make_eta_ode(base_state, m, k):

    # define the function eta_prime to give to solve_ivp and return it
    def eta_prime(R, eta):
        R = float(R)
        R = max(R, 1e-12) # ensuring R is not approximately 0, avoiding division errors

        uhat = eta[0:3].astype(complex)
        that = eta[3:6].astype(complex) # traction conditions

        # defining properties of the base state
        r, r_prime, region = base_state.base_state_params_given_R(R)

        # choose region params
        if region == "subcortex":
            params = base.subcortex_vals
        else:
            params = base.cortex_vals

        lambd = params["lambd"]
        mu = params["mu"]
        g_r = params["g_r"]
        g_theta = params["g_theta"]
        g_z = base.g_z

        # Base deformation gradient F_0 = diag(r', r/R, C_z)
        F_0 = np.diag([r_prime, r/R, base.C_z]).astype(complex)

        # Growth tensor Fg = diag(g_r, g_theta, g_z)
        Fg = np.diag([g_r, g_theta, g_z]).astype(complex)

        # STEP 1: reconstruct uhat' from traction relation
        uhat_prime = reconstruct_uhat_prime(R, uhat, that, F_0, Fg, lambd, mu, m, k)

        # STEP 2: compute full delta_P at this R
        dF = deltaF_hat(R, uhat, uhat_prime, m, k)
        dP = deltaP_hat(dF, F_0, Fg, lambd, mu)

        # Extract needed stress components for the Div equations
        dP_RR = dP[0, 0]
        dP_RTheta = dP[0, 1]
        dP_RZ = dP[0, 2]

        dP_ThetaR = dP[1, 0]
        dP_ThetaTheta = dP[1, 1]
        dP_ThetaZ = dP[1, 2]

        dP_ZR = dP[2, 0]
        dP_ZTheta = dP[2, 1]
        dP_ZZ = dP[2, 2]

        # STEP 3: equilibrium gives that', where that = [deltaP_RR, deltaP_ThetaR, deltaP_ZR]
        i = 1j # complex number

        delta_that_R = -(1/R)*(dP_RR - dP_ThetaTheta) - (i*m/R)*dP_RTheta - (i*k)*dP_RZ
        delta_that_Theta = -(1/R)*(dP_ThetaR + dP_RTheta) - (i*m/R)*dP_ThetaTheta - (i*k)*dP_ThetaZ
        delta_that_Z = -(1/R)*dP_ZR - (i*m/R)*dP_ZTheta - (i*k)*dP_ZZ

        eta_prime = np.zeros(6, dtype=complex)
        eta_prime[0:3] = uhat_prime
        eta_prime[3:6] = np.array([delta_that_R, delta_that_Theta, delta_that_Z], dtype=complex)

        return eta_prime

    return eta_prime

####################################
# Boundary condition in Stroh form #
####################################
def pressure_bc_operator(R_boundary, F_0, Fg, lambd, mu, m, k, P_f, N_R):

    # Build Q,R at boundary
    Q, W = stroh_QR_matrices(R_boundary, F_0, Fg, lambd, mu, m, k)
    Q_inv = inv(Q)

    # Build linear rhs terms
    S_0 = np.zeros((3, 3), dtype=complex)
    S_1 = np.zeros((3, 3), dtype=complex)

    uhat_0 = np.zeros(3, dtype=complex)
    uhat_prime_0 = np.zeros(3, dtype=complex)

    # Build S_0 for u' = 0
    for j in range(3):
        ej = np.zeros(3, dtype=complex)
        ej[j] = 1.0
        dF = deltaF_hat(R_boundary, ej, uhat_prime_0, m, k)
        S_0[:, j] = boundary_condition_rhs(dF, F_0, P_f, N_R=N_R)

    # Build S_1 for u = 0
    for j in range(3):
        ej = np.zeros(3, dtype=complex)
        ej[j] = 1.0
        dF = deltaF_hat(R_boundary, uhat_0, ej, m, k)
        S_1[:, j] = boundary_condition_rhs(dF, F_0, P_f, N_R=N_R)

    # pressure_rhs
    B_t = (N_R * np.eye(3, dtype=complex)) - (S_1 @ Q_inv)
    B_u = (S_1 @ Q_inv @ W) - S_0
    return B_u, B_t

################################
# Inner BC Matrices at R = R_v #
################################
def inner_bc_matrices(base_state, m, k):

    R = max(float(base.R_v), 1e-12)

    r, r_prime, _region = base_state.base_state_params_given_R(R)
    params = base.subcortex_vals
    lambd = params["lambd"]
    mu = params["mu"]
    g_r = params["g_r"]
    g_theta = params["g_theta"]
    g_z = base.g_z

    F_0 = np.diag([r_prime, r / R, base.C_z]).astype(complex)
    Fg  = np.diag([g_r, g_theta, g_z]).astype(complex)

    B_u, B_t = pressure_bc_operator(
        R_boundary=R, F_0=F_0, Fg=Fg, lambd=lambd, mu=mu,
        m=m, k=k, P_f=base.P_f, N_R=-1.0  # N_s = (-1,0,0)
    )
    return B_u, B_t

################################
# Outer BC residual at R = R_c #
################################
def outer_bc_residual(base_state, m, k, eta_at_R_c):
    R = base.R_c
    R = max(float(R), 1e-12)

    uhat = eta_at_R_c[0:3].astype(complex)
    that = eta_at_R_c[3:6].astype(complex)

    r, r_prime, region = base_state.base_state_params_given_R(R)
    params = base.cortex_vals
    lambd = params["lambd"]
    mu = params["mu"]
    g_r = params["g_r"]
    g_theta = params["g_theta"]
    g_z = base.g_z

    F_0 = np.diag([r_prime, r / R, base.C_z]).astype(complex)
    Fg = np.diag([g_r, g_theta, g_z]).astype(complex)

    B_u, B_t = pressure_bc_operator(
        R_boundary=R, F_0=F_0, Fg=Fg, lambd=lambd, mu=mu,
        m=m, k=k, P_f=base.P_f, N_R=+1.0 # N_c = (1,0,0)
    )

    return B_u @ uhat + B_t @ that

#######################################################################################
# Integrate Stroh state eta from R_v to R_c enforcing continuity at the interface R_s #
#######################################################################################
def integrate_eta(ode, eta_0, rtol=1e-6, atol=1e-9):

    R0 = base.R_v
    Rs = base.R_s
    Rc = base.R_c

    # use built in solver that uses the RK45 method to solve for eta in subcortex region
    sol1 = solve_ivp(fun=ode, t_span=(R0, Rs), y0=eta_0,
                     method="RK45", rtol=rtol, atol=atol)
    if not sol1.success:
        raise RuntimeError("IVP failed on [R_v, R_s]. " + sol1.message)

    eta_Rs = sol1.y[:, -1]

    # use built in solver that uses the RK45 method to solve for eta in subcortex region
    sol2 = solve_ivp(fun=ode, t_span=(Rs, Rc), y0=eta_Rs,
                     method="RK45", rtol=rtol, atol=atol)
    if not sol2.success:
        raise RuntimeError("IVP failed on [R_s, R_c]. " + sol2.message)

    eta_Rc = sol2.y[:, -1]
    return eta_Rc

#######################################
# Deriving the compatibility matrix B #
#######################################
def compatibility_matrix_B(g_theta_c, m, k, rtol=1e-6, atol=1e-9):

    base_state = solve_base_state_for_gthetac(g_theta_c)
    ode = make_eta_ode(base_state, m, k)

    B = np.zeros((3, 3), dtype=complex)

    # Choosing 3 independent initial states that are consistent with the inner BC
    if base.INNER_BC == "fixed":

        # Let u(R_v) = 0 and choose traction basis
        u_0 = np.zeros(3, dtype=complex)
        t_basis = np.eye(3, dtype=complex)

        initial_eta = []
        for j in range(3):
            t_0 = t_basis[:, j]
            initial_eta.append(np.concatenate([u_0, t_0]).astype(complex))

    elif base.INNER_BC == "pressure":

        # Let B_u*uhat + B_t*that = 0 at R=R_v and choose u-basis and solve for t
        B_u, B_t = inner_bc_matrices(base_state, m, k)
        u_basis = np.eye(3, dtype=complex)

        initial_eta = []
        for j in range(3):
            u_0 = u_basis[:, j]

            # Solve B_t*t_0 = -B_u*u_0
            t_0 = solve(B_t, -B_u @ u_0)
            initial_eta.append(np.concatenate([u_0, t_0]).astype(complex))

    else:
        raise ValueError("base.INNER_BC must be 'fixed' or 'pressure'.")

    # propagate each initial state and enforce outer BC
    for j in range(3):
        eta_0 = initial_eta[j]
        eta_Rc = integrate_eta(ode, eta_0, rtol=rtol, atol=atol)
        res = outer_bc_residual(base_state, m, k, eta_Rc)
        B[:, j] = res

    return B

################################
# Stability Indicator Function #
################################
def stability_indicators(B):
    singular_values = np.linalg.svd(B, compute_uv=False)
    s_min = float(singular_values[-1]) # smallest singular value
    s_max = float(singular_values[0]) # largest singular value
    relative_s_min = float(s_min/s_max)
    cond = float(singular_values[0] / singular_values[-1]) # condition number

    detB = np.linalg.det(B)
    absolute_val_of_detB = float(abs(detB))

    return absolute_val_of_detB, s_min, relative_s_min, cond


###############################
# Scanning g_theta_c function #
###############################
def scan_g_theta_c(g_min, g_max, n, m, k, rtol=1e-6, atol=1e-9):
    gs = np.linspace(g_min, g_max, n)

    absolute_val_of_detBs = np.full(n, np.nan)
    s_mins = np.full(n, np.nan)
    relative_s_mins = np.full(n, np.nan)
    conds = np.full(n, np.nan)

    for i, g in enumerate(gs):
        try:
            B = compatibility_matrix_B(
                g_theta_c=float(g),
                m=m,
                k=k,
                rtol=rtol,
                atol=atol
            )

            absolute_val_of_detB, s_min, relative_s_min, cond = stability_indicators(B)

            absolute_val_of_detBs[i] = absolute_val_of_detB
            relative_s_mins[i] = relative_s_min
            s_mins[i] = s_min
            conds[i] = cond

            print(
                f"g_theta_c={g:.8f}, "
                f"|det(B)|={absolute_val_of_detB:.3e}, "
                f"relative_s_min(B)={relative_s_min:.3e}, "
                f"s_min(B)={s_min:.3e}, "
                f"cond(B)={cond:.3e}"
            )

        except Exception as e:
            print(f"g_theta_c={g:.8f}: ERROR: {e}")

    return gs, absolute_val_of_detBs, relative_s_mins, s_mins, conds

##################################################
# Plotting det(B) and s_min(B) given a g_theta_c #
##################################################
def plot_g_theta_scan(gs, absolute_val_of_detBs, relative_s_mins, s_mins, m, k):
    import matplotlib.pyplot as plt

    finite_det = np.isfinite(absolute_val_of_detBs)
    finite_smin = np.isfinite(s_mins)

    plt.figure()
    plt.semilogy(gs[finite_det], absolute_val_of_detBs[finite_det], marker="o")
    plt.xlabel(r"$g_{\theta_c}$")
    plt.ylabel(r"$|\det(B)|$")
    plt.title(f"Compatibility determinant for mode m={m}, k={k}")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.semilogy(gs[finite_smin], relative_s_mins[finite_smin], marker="o")
    plt.xlabel(r"$g_{\theta_c}$")
    plt.ylabel(r"Relatively smallest singular value of $B$: relative_s_min($B$)")
    plt.title(f"Near-singularity indicator for mode m={m}, k={k}")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.semilogy(gs[finite_smin], s_mins[finite_smin], marker="o")
    plt.xlabel(r"$g_{\theta_c}$")
    plt.ylabel(r"Smallest singular value of $B$: s_min($B$)")
    plt.title(f"Near-singularity indicator for mode m={m}, k={k}")
    plt.grid(True)
    plt.show()


###############################
# Running the main simulation #
###############################
if __name__ == "__main__":
    m = 8
    k = 0.0

    g_min = 2.26355
    g_max = 2.26357
    n = 101

    gs, absolute_val_of_detBs, relative_s_mins, s_mins, conds = scan_g_theta_c(
        g_min=g_min,
        g_max=g_max,
        n=n,
        m=m,
        k=k
    )

    plot_g_theta_scan(gs, absolute_val_of_detBs, relative_s_mins, s_mins, m, k)

    best_idx = np.nanargmin(s_mins)

    print("\nBest near-buckling candidate in this scan:")
    print(f"  g_theta_c = {gs[best_idx]:.10f}")
    print(f"  |det(B)|  = {absolute_val_of_detBs[best_idx]:.3e}")
    print(f"  relative_s_min(B)  = {relative_s_mins[best_idx]:.3e}")
    print(f"  s_min(B)  = {s_mins[best_idx]:.3e}")
    print(f"  cond(B)  = {conds[best_idx]:.3e}")

