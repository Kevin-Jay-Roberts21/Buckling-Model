# Incremental stability analysis using a Stroh first-order system in R.
# Code uses radially_symmetric_solution_two_region.py

# Kevin Roberts
# January 2026

import numpy as np
from numpy.linalg import det, solve, inv
from scipy.integrate import solve_ivp

# import radially symmetric solver
import radially_symmetric_solution_two_region as base

###########################################################
# USING RADIALLY SYMMETRIC SOLUTION TO GET THE BASE STATE #
###########################################################
# For convenience, we write a "base state" wrapper as a class, getting all of the
# information from the radially symmetric solution
class BaseState:
    # Holds the piecewise base solution r(R), r'(R) from solve_bvp objects and
    # provides evaluation+region selection
    def __init__(self, subcortex_sol, cortex_sol):
        self.subcortex_sol = subcortex_sol
        self.cortex_sol = cortex_sol

    def eval(self, R):
        # returns (r, r_prime, region_name) at scalar R
        R = float(R)
        if R <= base.R_s:
            r, r_prime = self.subcortex_sol(np.array([R]))
            return float(r[0]), float(r_prime[0]), "subcortex"
        else:
            r, r_prime = self.cortex_sol.sol(np.array([R]))
            return float(r[0]), float(r_prime[0]), "cortex"

def solve_base_state_for_gthetac(g_theta_c):
    """
    Update cortex g_theta, solve the radial base state, and return a BaseState object.
    """
    # Update cortex growth multiplier in-place (radial solver reads cortex_vals)
    base.cortex_vals["g_theta"] = float(g_theta_c)

    # Solve the base state using the radial solver's function
    sub_sol, cor_sol, r_s_star = base.solve_base_state()

    return BaseState(sub_sol=sub_sol, cor_sol=cor_sol)


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

    # defining the delta F hat matrix and it's components
    dF_hat = np.zeros((3,3), dtype=complex)
    dF_hat[0, 0] = u_r_prime
    dF_hat[0, 1] = (i*m/R)*u_r - u_theta/R
    dF_hat[0, 2] = (i*k)*u_r
    dF_hat[1, 0] = u_theta_prime
    dF_hat[1, 1] = (i*m/R)*u_theta + u_r/R
    dF_hat[1, 2] = (i*k)*u_theta
    dF_hat[2, 0] = u_z_prime
    dF_hat[2, 1] = (i*m/R)*u_z
    dF_hat[2, 2] = (i*k)*u_z

    return dF_hat

#####################################
# Defining linearized, Piola stress #
#####################################
def deltaP_hat(dF, F_0, Fg, lambd, mu):

    # Fg matrix computations
    Fg_inverse = inv(Fg)
    Fg_inverse_transpose = Fg_inverse.T

    # Fe_0 matrix definition and computations
    Fe_0 = F_0 @ Fg_inverse.T # the symbol @ just compute matrix multiplication
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

##########################################################
# Defining the rhs of the linearized boundary conditions #
##########################################################
def incremental_pressure_rhs(dF, F_0, P_f, Nsign=+1.0):

    # defining the F_0 matrix computations
    F_0_inverse = inv(F_0)
    F_0_inverse_transpose = F_0_inverse.T
    J_0 = det(F_0)

    # defining the normal vector N
    N = np.array([Nsign, 0, 0], dtype=complex)

    rhs = -P_f*J_0*((np.trace(F_0_inverse@dF)*F_0_inverse_transpose - (F_0_inverse_transpose @ dF.T @ F_0_inverse_transpose)) @ N)

    return rhs

###############################################
# Build the Stroh-style ODE: eta' = f(R, eta) #
###############################################
def make_eta_ode(base_state, m, k):

    # define a function f to give to solve_ivp and return it
    def f(R, eta):
        R = float(R)
        uhat = eta[0:3].astype(complex)
        that = eta[3:6].astype(complex) # traction conditions

        # defining properties a the base state
        r, r_prime, region = base_state.eval(R)

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
        R = max(R, 1e-12)
        F_0 = np.diag([r_prime, r/R, base.C_z])

        # Growth tensor Fg = diag(g_r, g_theta, g_z)
        Fg = np.diag([g_r, g_theta, g_z])

        # STEP 1: reconstruct uhat' from traction relation
        # We need to solve that = B*uhat' + b0(uhat) b probing

        # b0 = traction when uhat' = 0
        uhat_prime0 = np.zeros(3, dtype=complex)
        dF_0 = deltaF_hat(R, uhat, uhat_prime0, m, k)
        dP_0 = deltaP_hat(dF_0, F_0, Fg, lambd, mu)
        b0 = np.array([dP_0[0, 0], dP_0[1, 0], dP_0[2, 0]], dtype=complex)

        # B columns: traction reponse to uhat' with uhat=0
        B = np.zeros((3, 3), dtype=complex)
        uhat_zero = np.zeros(3, dtype=complex)
        for j in range(3):
            uhat_prime = np.zeros(3, dtype=complex)
            uhat_prime[j] = 1
            dF = deltaF_hat(R, uhat_zero, uhat_prime, m, k)
            dP = deltaP_hat(dF, F_0, Fg, lambd, mu)
            B[:, j] = np.array([dP[0, 0], dP[1, 0], dP[2, 0]], dtype=complex)

        # solve for uhat'
        uhat_prime = solve(B, that - b0)


        # STEP 2: compute full delta_P at this R
        dF = deltaF_hat(R, uhat, uhat_prime, m, k)
        dP = deltaP_hat(dF, F_0, Fg, lambd, mu)

        # Extract needed stress components for the Div equations
        dP_RR = dP[0, 0]
        dP_ThetaTheta = dP[1, 1]
        dP_RTheta = dP[0, 1]
        dP_RZ = dP[0, 2]
        dP_ThetaR = dP[1, 0]
        dP_ThetaZ = [1, 2]
        dP_ZR = dP[2, 0]
        dP_ZTheta = dP[2, 1]
        dP_ZZ = dP[2, 2]

        # complex number
        i = 1j

        # STEP 3: equilibrium gives that', where that = [deltaP_RR, deltaP_ThetaR, deltaP_ZR]
        delta_that_R = -(1/R)*(dP_RR - dP_ThetaTheta) - (i*m/R)*dP_RTheta - (i*k)*dP_RZ
        delta_that_Theta = -(1/R)*(dP_ThetaR + dP_RTheta) - (i*m/R)*dP_ThetaTheta - (i*k)*dP_ThetaZ
        delta_that_Z = -(1/R)*dP_ZR - (i*m/R)*dP_ZTheta - (i*k)*dP_ZZ

        eta_prime = np.zeros(6, dtype=complex)
        eta_prime[0:3] = uhat_prime
        eta_prime[3:6] = np.array([delta_that_R, delta_that_Theta, delta_that_Z], dtype=complex)

        return eta_prime

    return f

################################
# Outer BC residual at R = R_c #
################################
def outer_bc_residual(base_state, m, k, eta_at_R_c):
    R = base.R_c
    uhat = eta_at_R_c[0:3].astype(complex)
    that = eta_at_R_c[3:6].astype(complex)

    r, r_prime, region = base_state.eval(R)
    params = base.cortex_vals
    lambd = params["lambd"]
    mu = params["mu"]
    g_r = params["g_r"]
    g_theta = params["g_theta"]
    g_z = base.g_z

    R = max(float(R), 1e-12)
    F_0 = np.diag([r_prime, r/R, base.C_z])
    Fg = np.diag([g_r, g_theta, g_z])

    # reconstruct uhat' same way as inside the ODE
    uhat_prime0 = np.zeros(3, dtype=complex)
    dF_0 = deltaF_hat(R, uhat, uhat_prime0, m, k)
    dP_0 = deltaP_hat(dF_0, F_0, Fg, lambd, mu)
    b0 = np.array([dP_0[0, 0], dP_0[1, 0], dP_0[2, 0]], dtype=complex)

    B = np.zeros((3, 3), dtype=complex)
    uhat_zero = np.zeros(3, dtype=complex)
    for j in range(3):
        uhat_prime = np.zeros(3, dtype=complex)
        uhat_prime[j] = 1
        dF = deltaF_hat(R, uhat_zero, uhat_prime, m, k)
        dP = deltaP_hat(dF, F_0, Fg, lambd, mu)
        B[:, j] = np.array([dP[0, 0], dP[1, 0], dP[2, 0]], dtype=complex)

    uhat_prime = solve(B, that - b0)

    # compute deltaF for BC RHS
    dF = deltaF_hat(R, uhat, uhat_prime, m, k)

    # RHS traction from incremental pressure BC
    rhs = incremental_pressure_rhs(dF, F_0, base.P_f, Nsign=+1.0)

    # return the residual
    return that - rhs

#####################################################################
# Deriving the shooting matrix and stability indicator Φ = |det(S)| #
#####################################################################
def shooting_matrix_S(g_theta_c, m, k, rtol=1e-6, atol=1e-9):
    # Building the 3x3 shooting matrix S for given g_theta_c, m, k
    # The columns correspond to three independent inner traction initializations.
    base_state = solve_base_state_for_gthetac(g_theta_c)
    ode = make_eta_ode(base_state, m, k)

    R_0 = base.R_v
    R_f = base.R_c

    # inner fixed displacement condition: uhat(R_v)=0
    u_0 = np.zeros(3, dtype=complex)

    # basis tractions at inner boundary
    E = np.eye(3, dtype=complex)

    S = np.zeros((3,3), dtype=complex)

    for j in range(3):
        t_0 = E[:, j]
        eta_0 = np.concatenate([u_0, t_0]).astype(complex)

        sol = solve_ivp(fun=lambda R, y: ode(R, y),
                        t_span=(R_0, R_f),
                        y_0 = eta_0,
                        method="RK45",
                        rtol=rtol,
                        atol=atol)

        if not sol.success:
            raise RuntimeError("IVP failed for g_theta_c=%g (m=%d,k=%g)" % (g_theta_c, m, k))

        eta_R_c = sol.y[:, -1]
        res = outer_bc_residual(base_state, m, k, eta_R_c)
        S[:, j] = res

    return S

def Phi(g_theta_c, m, k):
    # Stability indicator
    S = shooting_matrix_S(g_theta_c, m, k)
    return np.abs(det(S))

#########################################
# Scan utility for g_theta_c thresholds #
#########################################
def scan_g_theta_c(g_min, g_max, n, m, k):
    gs = np.linspace(g_min, g_max, n)
    vals = []

    for g in gs:
        try:
            vals.append(Phi(g, m, k))
            print(f"g_theta_c={g:.6f}, Phi={vals[-1]:.3e}")
        except Exception as e:
            vals.append(np.nan)
            print(f"g_theta_c={g:.6f}, Phi=nan  (error: {e})")
        return gs, np.array(vals)


if __name__ == "__main__":
    # Choose a mode to start with
    m = 8
    k = 0.0

    # Scan range for g_theta_c
    g_min = 1.00
    g_max = 2.00
    n = 11

    gs, phis = scan_g_theta_c(g_min, g_max, n, m, k)

    # crude "best guess" of critical g from scan
    j = np.nanargmin(phis)
    print("\nBest scan candidate:")
    print(f"  g_theta_c ≈ {gs[j]:.6f}, Phi ≈ {phis[j]:.3e}")