# Incremental stability analysis using a Stroh first-order system in R.
# Code uses radially_symmetric_solution_two_region.py

# Kevin Roberts
# January 2026

import numpy as np
from numpy.linalg import det, solve, inv
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

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
            r, r_prime = self.subcortex_sol.sol(np.array([R]))
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
    Fe_0 = F_0 @ Fg_inverse # the symbol @ just compute matrix multiplication
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
def incremental_pressure_rhs(dF, F_0, P_f, Nsign=+1.0):

    # defining the F_0 matrix computations
    F_0_inverse = inv(F_0)
    F_0_inverse_transpose = F_0_inverse.T
    J_0 = det(F_0)

    # defining the normal vector N
    N = np.array([Nsign, 0, 0], dtype=complex)

    rhs = -P_f*J_0*((np.trace(F_0_inverse@dF)*F_0_inverse_transpose - (F_0_inverse_transpose @ dF.T @ F_0_inverse_transpose)) @ N)

    return rhs

###############################################################
# Stroh building blocks: t = Qu' + Ru, with u' = Q^{-1}(t-Ru) #
###############################################################
def stroh_QR_matrices(R, F_0, Fg, lambd, mu, m, k):
    Rmat = np.zeros((3, 3), dtype=complex)
    up0 = np.zeros(3, dtype=complex)
    for j in range(3):
        ej = np.zeros(3, dtype=complex)
        ej[j] = 1.0
        dF = deltaF_hat(R, ej, up0, m, k)
        dP = deltaP_hat(dF, F_0, Fg, lambd, mu)
        Rmat[:, j] = np.array([dP[0, 0], dP[1, 0], dP[2, 0]], dtype=complex)

    Qmat = np.zeros((3, 3), dtype=complex)
    u0 = np.zeros(3, dtype=complex)
    for j in range(3):
        ej = np.zeros(3, dtype=complex)
        ej[j] = 1.0
        dF = deltaF_hat(R, u0, ej, m, k)
        dP = deltaP_hat(dF, F_0, Fg, lambd, mu)
        Qmat[:, j] = np.array([dP[0, 0], dP[1, 0], dP[2, 0]], dtype=complex)

    return Qmat, Rmat

def reconstruct_uhat_prime(R, uhat, that, F_0, Fg, lambd, mu, m, k):
    """
    Using Stroh: that = Q u' + R u  => u' = Q^{-1}(that - R u).
    """
    Qmat, Rmat = stroh_QR_matrices(R, F_0, Fg, lambd, mu, m, k)
    uhat_prime = solve(Qmat, that - Rmat @ uhat)
    return uhat_prime


###############################################
# Build the Stroh-style ODE: eta' = f(R, eta) #
###############################################
def make_eta_ode(base_state, m, k):

    # define a function f to give to solve_ivp and return it
    def f(R, eta):
        R = float(R)
        R = max(R, 1e-12)

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
        dP_ThetaTheta = dP[1, 1]
        dP_RTheta = dP[0, 1]

        dP_RZ = dP[0, 2]
        dP_ThetaR = dP[1, 0]
        dP_ThetaZ = dP[1, 2]

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

#####################################################
# Boundary condition in Stroh form: Bu u + Bt t = 0 #
#####################################################
def pressure_bc_operator(BR, F_0, Fg, lambd, mu, m, k, P_f, Nsign):

    # Build Q,R at boundary
    Qmat, Rmat = stroh_QR_matrices(BR, F_0, Fg, lambd, mu, m, k)
    Qinv = inv(Qmat)

    # Helper: given (u,t), compute rhs traction vector (3,) using your pressure RHS
    def rhs_from_ut(u, t):
        # u' = Q^{-1}(t - R u)
        up = Qinv @ (t - Rmat @ u)
        dF = deltaF_hat(BR, u, up, m, k)
        rhs = incremental_pressure_rhs(dF, F_0, P_f, Nsign=Nsign)
        return rhs

    # Build linear maps Mu, Mt such that rhs = Mu u + Mt t
    Mu = np.zeros((3, 3), dtype=complex)
    Mt = np.zeros((3, 3), dtype=complex)

    # rhs response to unit u (with t=0)
    t0 = np.zeros(3, dtype=complex)
    for j in range(3):
        ej = np.zeros(3, dtype=complex)
        ej[j] = 1.0
        Mu[:, j] = rhs_from_ut(ej, t0)

    # rhs response to unit t (with u=0)
    u0 = np.zeros(3, dtype=complex)
    for j in range(3):
        ej = np.zeros(3, dtype=complex)
        ej[j] = 1.0
        Mt[:, j] = rhs_from_ut(u0, ej)

    # Residual is: t - rhs = t - (Mu u + Mt t) = (-Mu)u + (I - Mt)t
    Bu = -Mu
    Bt = np.eye(3, dtype=complex) - Mt
    return Bu, Bt


################################
# Outer BC residual at R = R_c #
################################
def outer_bc_residual(base_state, m, k, eta_at_R_c):
    R = base.R_c
    R = max(float(R), 1e-12)

    uhat = eta_at_R_c[0:3].astype(complex)
    that = eta_at_R_c[3:6].astype(complex)

    r, r_prime, region = base_state.eval(R)
    params = base.cortex_vals
    lambd = params["lambd"]
    mu = params["mu"]
    g_r = params["g_r"]
    g_theta = params["g_theta"]
    g_z = base.g_z

    F_0 = np.diag([r_prime, r / R, base.C_z]).astype(complex)
    Fg = np.diag([g_r, g_theta, g_z]).astype(complex)

    Bu, Bt = pressure_bc_operator(
        BR=R, F_0=F_0, Fg=Fg, lambd=lambd, mu=mu,
        m=m, k=k, P_f=base.P_f, Nsign=+1.0
    )

    return Bu @ uhat + Bt @ that

#####################################################################
# Deriving the shooting matrix and stability indicator Φ = |det(S)| #
#####################################################################
def integrate_eta(base_state, ode, eta_0, rtol=1e-6, atol=1e-9):
    """
    Integrate in two pieces [R_v,R_s] and [R_s,R_c] to match the paper narrative.
    """
    R0 = base.R_v
    Rs = base.R_s
    Rc = base.R_c

    sol1 = solve_ivp(fun=ode, t_span=(R0, Rs), y0=eta_0,
                     method="RK45", rtol=rtol, atol=atol)
    if not sol1.success:
        raise RuntimeError("IVP failed on [R_v,R_s]. " + sol1.message)

    eta_Rs = sol1.y[:, -1]

    sol2 = solve_ivp(fun=ode, t_span=(Rs, Rc), y0=eta_Rs,
                     method="RK45", rtol=rtol, atol=atol)
    if not sol2.success:
        raise RuntimeError("IVP failed on [R_s,R_c]. " + sol2.message)

    eta_Rc = sol2.y[:, -1]
    return eta_Rc

#####################################################################
# Deriving the shooting matrix and stability indicator Φ = |det(S)| #
#####################################################################
def shooting_matrix_S(g_theta_c, m, k, rtol=1e-6, atol=1e-9):
    # Building the 3x3 shooting matrix S for given g_theta_c, m, k
    # The columns correspond to three independent inner traction initializations.
    base_state = solve_base_state_for_gthetac(g_theta_c)
    ode = make_eta_ode(base_state, m, k)

    # inner fixed displacement condition: uhat(R_v)=0
    u_0 = np.zeros(3, dtype=complex)

    # basis tractions at inner boundary
    E = np.eye(3, dtype=complex)

    S = np.zeros((3, 3), dtype=complex)

    for j in range(3):
        t_0 = E[:, j]
        eta_0 = np.concatenate([u_0, t_0]).astype(complex)

        # ---- key change: integrate in two pieces using your helper ----
        eta_R_c = integrate_eta(base_state, ode, eta_0, rtol=rtol, atol=atol)

        # apply outer BC residual at R=R_c
        res = outer_bc_residual(base_state, m, k, eta_R_c)
        S[:, j] = res

    return S

def stability_indicators(S):
    svals = np.linalg.svd(S, compute_uv=False)
    smin = float(svals[-1])
    cond = float(svals[0] / svals[-1])
    # log10|det| is less insane than |det|
    sign, logabsdet = np.linalg.slogdet(S)
    log10absdet = float(logabsdet / np.log(10.0))
    return smin, cond, log10absdet

def Phi(g_theta_c, m, k):
    S = shooting_matrix_S(g_theta_c, m, k)
    smin, cond, log10absdet = stability_indicators(S)
    return smin, cond, log10absdet


#########################################
# Scan utility for g_theta_c thresholds #
#########################################
def scan_g_theta_c(g_min, g_max, n, m, k):
    gs = np.linspace(g_min, g_max, n)

    phis = np.empty(gs.shape, dtype=float)
    conds = np.empty(gs.shape, dtype=float)

    for i, g in enumerate(gs):
        try:
            smin, condS, log10det = Phi(g, m, k)
            phis[i] = float(smin)
            conds[i] = float(condS)
            print(f"g={g:.6f}, smin={smin:.3e}, cond={condS:.3e}, log10|det|={log10det:.2f}")
        except Exception as e:
            phis[i] = np.nan
            conds[i] = np.nan
            print(f"g={g:.6f}: ERROR: {e}")

    return gs, phis, conds

def find_critical_g_theta_c(
    g_min, g_max, m, k,
    n_scan=401,
    smin_trigger=1e-4,
    refine_pad=2,
    xatol=1e-10,
    maxiter=200,
    verbose=True
):
    """
    Find the FIRST critical g_theta_c in [g_min, g_max] for fixed (m,k).

    Strategy:
      1) coarse scan of smin(g)
      2) find first local minimum with smin < smin_trigger (if none, use global min)
      3) refine that candidate with bounded minimization on a small interval
      4) return gcrit and diagnostic info

    Returns:
      gcrit (float), info (dict)
    """

    gs = np.linspace(g_min, g_max, n_scan)
    smins = np.full(gs.shape, np.nan, dtype=float)
    conds = np.full(gs.shape, np.nan, dtype=float)

    # ---- coarse scan ----
    for i, g in enumerate(gs):
        try:
            smin, condS, _log10det = Phi(float(g), m, k)
            smins[i] = float(smin)
            conds[i] = float(condS)
            if verbose:
                print(f"[scan] g={g:.6f}, smin={smins[i]:.3e}, cond={conds[i]:.3e}")
        except Exception as e:
            if verbose:
                print(f"[scan] g={g:.6f}: ERROR: {e}")

    if np.all(np.isnan(smins)):
        raise RuntimeError("All scan values were NaN; base state / IVP failing across [g_min,g_max].")

    # ---- local minima indices ----
    local_min_idx = []
    for i in range(1, len(gs) - 1):
        if np.isfinite(smins[i-1:i+2]).all():
            if smins[i] <= smins[i-1] and smins[i] <= smins[i+1]:
                local_min_idx.append(i)

    # ---- pick FIRST local min below trigger ----
    cand_idx = None
    for i in local_min_idx:
        if smins[i] < smin_trigger:
            cand_idx = i
            break

    # fallback: global minimum over scan
    if cand_idx is None:
        cand_idx = int(np.nanargmin(smins))

    j = cand_idx
    j0 = max(j - refine_pad, 0)
    j1 = min(j + refine_pad, len(gs) - 1)
    a, b = float(gs[j0]), float(gs[j1])

    if verbose:
        print("\n[refine] scan candidate:")
        print(f"  g≈{gs[j]:.12f}, smin≈{smins[j]:.3e}, cond≈{conds[j]:.3e}")
        print(f"  refine interval: [{a:.12f}, {b:.12f}]")

    # ---- bounded refine ----
    def smin_of_g(g):
        S = shooting_matrix_S(float(g), m, k)
        svals = np.linalg.svd(S, compute_uv=False)
        return float(svals[-1])

    res = minimize_scalar(
        smin_of_g,
        bounds=(a, b),
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter}
    )

    gcrit = float(res.x)

    # ---- verification at gcrit ----
    Scrit = shooting_matrix_S(gcrit, m, k)
    U, svals, Vh = np.linalg.svd(Scrit)
    v_null = Vh[-1, :].conj().T
    smin_crit = float(svals[-1])
    cond_crit = float(svals[0] / svals[-1])

    info = {
        "gs_scan": gs,
        "smins_scan": smins,
        "conds_scan": conds,
        "scan_candidate_g": float(gs[j]),
        "scan_candidate_smin": float(smins[j]),
        "scan_candidate_cond": float(conds[j]),
        "refine_interval": (a, b),
        "opt_success": bool(res.success),
        "opt_message": str(res.message),
        "opt_fun": float(res.fun),
        "gcrit": gcrit,
        "Scrit": Scrit,
        "svals_crit": svals,
        "smin_crit": smin_crit,
        "cond_crit": cond_crit,
        "v_null": v_null,
        "Sv_norm": float(np.linalg.norm(Scrit @ v_null)),
    }

    if verbose:
        print("\n[result] estimated critical g_theta_c:")
        print(f"  gcrit = {gcrit:.12f}")
        print(f"  svals = {svals}")
        print(f"  smin  = {smin_crit:.3e}")
        print(f"  cond  = {cond_crit:.3e}")
        print(f"  ||S v|| = {info['Sv_norm']:.3e}")
        print(f"  v_null = {v_null}")

    return gcrit, info




if __name__ == "__main__":
    # Choose a mode
    m = 8
    k = 0.0

    # Global search range
    g_min = 1.0
    g_max = 3.0

    # 1) Find critical using scan + bounded refine
    gcrit, info = find_critical_g_theta_c(
        g_min=g_min, g_max=g_max,
        m=m, k=k,
        n_scan=401,          # coarse scan resolution
        smin_trigger=1e-4,   # adjust if needed
        refine_pad=2,
        xatol=1e-10,
        verbose=True
    )

    # 2) (Optional) print a small summary
    print("\nSUMMARY")
    print(f"  gcrit ≈ {gcrit:.12f}")
    print(f"  smin(gcrit) ≈ {info['smin_crit']:.3e}")
    print(f"  cond(S(gcrit)) ≈ {info['cond_crit']:.3e}")

    # 3) (Optional) keep your old scan printout for comparison
    n = 41
    gs, phis, conds = scan_g_theta_c(g_min, g_max, n, m, k)

    if np.all(np.isnan(phis)):
        raise RuntimeError("All Phi values were NaN. See scan errors above.")

    j = np.nanargmin(phis)
    print("\nBest scan grid point (coarse):")
    print(f"  g_theta_c ≈ {gs[j]:.6f}, smin ≈ {phis[j]:.3e}, cond(S) ≈ {conds[j]:.3e}")

    # 4) Verify at the refined gcrit too (already in info, but here’s the print)
    Scrit = info["Scrit"]
    svals = info["svals_crit"]
    print("\nSVD at refined gcrit:")
    print("  singular values:", svals)
    print("  cond:", svals[0] / svals[-1])

