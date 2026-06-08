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

    def rhs_from_ut(u, t):
        # u' = Q^{-1}(t - R u)
        up = Qinv @ (t - Rmat @ u)
        dF = deltaF_hat(BR, u, up, m, k)
        rhs = incremental_pressure_rhs(dF, F_0, P_f, Nsign=Nsign)
        return rhs

    # Build linear maps Mu, Mt such that rhs = Mu u + Mt t
    Mu = np.zeros((3, 3), dtype=complex)
    Mt = np.zeros((3, 3), dtype=complex)

    t0 = np.zeros(3, dtype=complex)
    for j in range(3):
        ej = np.zeros(3, dtype=complex)
        ej[j] = 1.0
        Mu[:, j] = rhs_from_ut(ej, t0)

    u0 = np.zeros(3, dtype=complex)
    for j in range(3):
        ej = np.zeros(3, dtype=complex)
        ej[j] = 1.0
        Mt[:, j] = rhs_from_ut(u0, ej)

    # Residual: (deltaP*N) - rhs = (Nsign*t) - (Mu u + Mt t)
    # => (-Mu) u + (Nsign*I - Mt) t
    Bu = -Mu
    Bt = (Nsign * np.eye(3, dtype=complex)) - Mt
    return Bu, Bt

def inner_bc_matrices(base_state, m, k):

    R = max(float(base.R_v), 1e-12)

    r, r_prime, _region = base_state.eval(R)
    params = base.subcortex_vals
    lambd = params["lambd"]
    mu = params["mu"]
    g_r = params["g_r"]
    g_theta = params["g_theta"]
    g_z = base.g_z

    F_0 = np.diag([r_prime, r / R, base.C_z]).astype(complex)
    Fg  = np.diag([g_r, g_theta, g_z]).astype(complex)

    Bu, Bt = pressure_bc_operator(
        BR=R, F_0=F_0, Fg=Fg, lambd=lambd, mu=mu,
        m=m, k=k, P_f=base.P_f, Nsign=-1.0  # N_s = (-1,0,0)
    )
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

##########################################################################
# Deriving the compatibility matrix and stability indicator Φ = |det(B)| #
##########################################################################
def compatibility_matrix_B(g_theta_c, m, k, rtol=1e-6, atol=1e-9):

    base_state = solve_base_state_for_gthetac(g_theta_c)
    ode = make_eta_ode(base_state, m, k)

    B = np.zeros((3, 3), dtype=complex)

    # --- choose 3 independent initial states consistent with inner BC ---
    if base.INNER_BC == "fixed":
        # u(R_v) = 0, choose traction basis
        u0 = np.zeros(3, dtype=complex)
        Tbasis = np.eye(3, dtype=complex)

        initials = []
        for j in range(3):
            t0 = Tbasis[:, j]
            initials.append(np.concatenate([u0, t0]).astype(complex))

    elif base.INNER_BC == "pressure":
        # Bu u + Bt t = 0 at R=R_v, choose u-basis and solve for t
        Bu, Bt = inner_bc_matrices(base_state, m, k)

        Ubasis = np.eye(3, dtype=complex)
        initials = []
        for j in range(3):
            u0 = Ubasis[:, j]
            # Solve Bt t0 = -Bu u0
            t0 = solve(Bt, -Bu @ u0)
            initials.append(np.concatenate([u0, t0]).astype(complex))

    else:
        raise ValueError("base.INNER_BC must be 'fixed' or 'pressure'.")

    # --- propagate each admissible initial state, enforce outer BC ---
    for j in range(3):
        eta_0 = initials[j]
        eta_Rc = integrate_eta(base_state, ode, eta_0, rtol=rtol, atol=atol)
        res = outer_bc_residual(base_state, m, k, eta_Rc)
        B[:, j] = res

    return B

def stability_indicators(B):
    svals = np.linalg.svd(B, compute_uv=False)
    smin = float(svals[-1])
    cond = float(svals[0] / svals[-1])
    # log10|det| is less insane than |det|
    sign, logabsdet = np.linalg.slogdet(B)
    log10absdet = float(logabsdet / np.log(10.0))
    return smin, cond, log10absdet

def Phi(g_theta_c, m, k):
    B = compatibility_matrix_B(g_theta_c, m, k)
    smin, cond, log10absdet = stability_indicators(B)
    return smin, cond, log10absdet

# todo: Need to review this
def classify_buckling_status(smin, condB,
                             buckling_tol=1e-6,
                             candidate_tol=1e-1,
                             cond_candidate_tol=1e5):
    if smin < buckling_tol:
        return "LIKELY BUCKLING"
    elif smin < candidate_tol and condB > cond_candidate_tol:
        return "CANDIDATE BUCKLING — refine locally"
    else:
        return "NO CLEAR BUCKLING"


def scan_growth_with_buckling_flag(
    g_min, g_max, n, m, k,
    buckling_tol=1e-6,
    candidate_tol=1e-1,
    cond_candidate_tol=1e5
):
    gs = np.linspace(g_min, g_max, n)

    smins = np.full(n, np.nan)
    conds = np.full(n, np.nan)
    logdets = np.full(n, np.nan)
    statuses = np.empty(n, dtype=object)

    for i, g in enumerate(gs):
        try:
            smin, condB, log10det = Phi(float(g), m, k)

            smins[i] = smin
            conds[i] = condB
            logdets[i] = log10det

            status = classify_buckling_status(
                smin=smin,
                condB=condB,
                buckling_tol=buckling_tol,
                candidate_tol=candidate_tol,
                cond_candidate_tol=cond_candidate_tol
            )

            statuses[i] = status

            print(
                f"g={g:.6f}, "
                f"smin(B)={smin:.3e}, "
                f"cond(B)={condB:.3e}, "
                f"log10|det(B)|={log10det:.2f}, "
                f"status={status}"
            )

        except Exception as e:
            statuses[i] = "ERROR"
            print(f"g={g:.6f}: ERROR: {e}")

    flags = np.array([status == "LIKELY BUCKLING" for status in statuses])
    candidates = np.array(["CANDIDATE" in str(status) for status in statuses])

    return gs, smins, conds, logdets, flags, candidates, statuses

# todo: need to review this
def plot_buckling_scan(gs, smins, conds, flags, m, k):
    import matplotlib.pyplot as plt

    finite = np.isfinite(smins)

    plt.figure()
    plt.semilogy(gs[finite], smins[finite], marker="o")
    plt.xlabel(r"$g_{\theta_c}$")
    plt.ylabel(r"Smallest singular value of $B$")
    plt.title(f"Buckling indicator for mode m={m}, k={k}")
    plt.grid(True)

    if np.any(flags):
        g_first = gs[np.where(flags)[0][0]]
        plt.axvline(g_first, linestyle="--", label=f"first flagged g={g_first:.6f}")
        plt.legend()

    plt.show()

    plt.figure()
    plt.semilogy(gs[finite], conds[finite], marker="o")
    plt.xlabel(r"$g_{\theta_c}$")
    plt.ylabel(r"Condition number of $B$")
    plt.title(f"Condition number for mode m={m}, k={k}")
    plt.grid(True)

    if np.any(flags):
        g_first = gs[np.where(flags)[0][0]]
        plt.axvline(g_first, linestyle="--", label=f"first flagged g={g_first:.6f}")
        plt.legend()

    plt.show()


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
        B = compatibility_matrix_B(float(g), m, k)
        svals = np.linalg.svd(B, compute_uv=False)
        return float(svals[-1])

    res = minimize_scalar(
        smin_of_g,
        bounds=(a, b),
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter}
    )

    gcrit = float(res.x)

    # ---- verification at gcrit ----
    Bcrit = compatibility_matrix_B(gcrit, m, k)
    U, svals, Vh = np.linalg.svd(Bcrit)
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
        "Bcrit": Bcrit,
        "svals_crit": svals,
        "smin_crit": smin_crit,
        "cond_crit": cond_crit,
        "v_null": v_null,
        "Bv_norm": float(np.linalg.norm(Bcrit @ v_null)),
    }

    if verbose:
        print("\n[result] estimated critical g_theta_c:")
        print(f"  gcrit = {gcrit:.12f}")
        print(f"  svals = {svals}")
        print(f"  smin  = {smin_crit:.3e}")
        print(f"  cond  = {cond_crit:.3e}")
        print(f"  ||B v|| = {info['Bv_norm']:.3e}")
        print(f"  v_null = {v_null}")

    return gcrit, info



# todo: need to review this
if __name__ == "__main__":
    m = 14
    k = 0.0

    g_min = 2.460167
    g_max = 2.460189
    n = 181

    buckling_tol = 1e-6
    candidate_tol = 1e-1
    cond_candidate_tol = 1e5

    gs, smins, conds, logdets, flags, candidates, statuses = scan_growth_with_buckling_flag(
        g_min=g_min,
        g_max=g_max,
        n=n,
        m=m,
        k=k,
        buckling_tol=buckling_tol,
        candidate_tol=candidate_tol,
        cond_candidate_tol=cond_candidate_tol
    )

    plot_buckling_scan(gs, smins, conds, flags | candidates, m, k)

    if np.any(flags):
        idx = np.where(flags)[0][0]
        print("\nLIKELY BUCKLING DETECTED")
        print(f"  first flagged g_theta_c = {gs[idx]:.10f}")
        print(f"  smin(B) = {smins[idx]:.3e}")
        print(f"  cond(B) = {conds[idx]:.3e}")

    elif np.any(candidates):
        idx = np.where(candidates)[0][0]
        best_idx = np.nanargmin(smins)

        print("\nCANDIDATE BUCKLING DETECTED — refine locally")
        print(f"  first candidate g_theta_c = {gs[idx]:.10f}")
        print(f"  best candidate g_theta_c = {gs[best_idx]:.10f}")
        print(f"  min smin(B) = {smins[best_idx]:.3e}")
        print(f"  cond(B) = {conds[best_idx]:.3e}")
        print("\nSuggested refined scan:")
        print(f"  g_min = {gs[best_idx] - 0.5:.6f}")
        print(f"  g_max = {gs[best_idx] + 0.5:.6f}")
        print("  n = 301")

    else:
        idx = np.nanargmin(smins)
        print("\nNO CLEAR BUCKLING DETECTED IN THIS RANGE")
        print("Smallest value found was only the best dip, not necessarily a threshold:")
        print(f"  best g_theta_c = {gs[idx]:.10f}")
        print(f"  min smin(B) = {smins[idx]:.3e}")
        print(f"  cond(B) = {conds[idx]:.3e}")

