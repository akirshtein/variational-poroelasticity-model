"""Discrete energy-law behaviour on the sealed blob problem (Figure 3 of the paper).

Two experiments, both on the unit square with all four walls clamped (u = 0 on the
whole boundary) and no external forcing, so the only energy exchange is Darcy
dissipation. The total no-flux condition j.n = 0 holds automatically as the natural
boundary condition of the density equation.

  timeseries   K = 32, dt = 1e-3, T = 0.5.  Records the normalised stored energy
               E(t)/E(0) and produces Figure 3 (left).
               -> data/energy_blob/energy_vs_time.csv

  defect       K = 64, T = 0.5, dt in {0.05, 0.02, 0.01, 0.005}.  Computes the
               cumulative energy-law discrepancy

                   DEFECT(T) = | E(T) - E(0) + dt * sum_n D_pred^n |,

               where D_pred^n = kappa * int |d-hat^{n+1/2}|^2 / rho-bar^{n+1/2} is
               exactly the dissipation term appearing in the discrete energy identity,
               so this tests that identity directly rather than through a proxy.
               Expected O(dt^2), from the O(dt^3) per-step defect.
               -> data/energy_blob/defect_vs_dt.csv

Both experiments start from zero displacement and the smooth density perturbation

    rho_f(x, 0) = 1 + 0.2 sin(pi x) sin(pi y),

equivalently p(x, 0) = 1 + 0.2 sin(pi x) sin(pi y) under the ideal-gas closure
p = M rho_f with M = 1.

Usage:  python run_energy_blob.py [timeseries|defect|all]      (default: all)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from porousflow import *
import ufl

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "energy_blob")
ensure_dir(BASE)

pi = np.pi

# Benchmark parameters (Section 4.1): E = 1, nu = 0.2  ->  mu = 5/12, lambda = 5/18.
MU, LAM, KAPPA, M_VAL = 5/12, 5/18, 1.0, 1.0


def rho_init_fn(xp):
    return 1.0 + 0.2*np.sin(pi*xp[0])*np.sin(pi*xp[1])


def build(K, dt, tag):
    """Assemble the two-field blob problem and return (problem, state, forms)."""
    pde = PorousFlow(mu=MU, lam=LAM, a=1.0, k=KAPPA, M=M_VAL, p=1)
    pde.set_parameters(dt=dt, K=K, T=1.0, formulation="twofield", ptype="fieldsplit",
                       FrequencyOutput=1, cell_type=CellType.triangle)
    pde.setup_spaces(build_boundary=False)
    u_tr, rho_tr, u_te, rho_te = pde.test_and_trial_functions()

    u        = Function(pde.spaces.Uspace)
    rho      = Function(pde.spaces.Pspace)
    u_prev   = Function(pde.spaces.Uspace)
    rho_prev = Function(pde.spaces.Pspace)

    rho.interpolate(rho_init_fn); rho_prev.interpolate(rho_init_fn)
    noslip = np.zeros(pde.spaces.mesh.geometry.dim, dtype=PETSc.ScalarType)
    U_bc = dirichletbc(noslip, pde.spaces.Uspace_boundary_dofs, pde.spaces.Uspace)
    bcs = [U_bc]
    pde.initialize_twofield_state(u, rho, u_prev, rho_prev, [U_bc])

    dt_c = Constant(pde.spaces.mesh, PETSc.ScalarType(dt))
    half = Constant(pde.spaces.mesh, PETSc.ScalarType(0.5))
    stab = Constant(pde.spaces.mesh, PETSc.ScalarType(dt))   # alpha_stab = dt

    avg_rho  = half*(rho + rho_prev)
    avg_domg = (avg_rho*grad(pde.domega(rho_prev))
                + rho_prev*grad(pde.d2omega(rho_prev)*(rho - rho_prev)))

    R_u   = (half*inner(pde.sigma((1+stab)*u + (1-stab)*u_prev), grad(u_te))*dx
             + dot(avg_domg, pde.a*u_te)*dx)
    R_rho = (dot(avg_domg, dt_c*pde.k*grad(rho_te))*dx
             + dot(-pde.a*avg_rho*(u - u_prev), grad(rho_te))*dx
             + (rho - rho_prev)*rho_te*dx)

    J11 = ufl.derivative(R_u,   u,   u_tr);  J12 = ufl.derivative(R_u,   rho, rho_tr)
    J21 = ufl.derivative(R_rho, u,   u_tr);  J22 = ufl.derivative(R_rho, rho, rho_tr)

    problem = NonlinearProblem([R_u, R_rho], [u, rho], bcs=bcs,
        J=[[J11,J12],[J21,J22]], P=[[J11,None],[None,J22]], kind="nest",
        petsc_options_prefix=f"blob_{tag}_",
        petsc_options={"snes_type":"newtonls","snes_rtol":1e-9,"snes_atol":1e-12,
                       "snes_max_it":60,"ksp_type":"gmres","ksp_rtol":1e-8,
                       "ksp_atol":1e-12,"pc_type":"fieldsplit"})
    ksp = problem.solver.getKSP(); pc = ksp.getPC()
    pc.setType("fieldsplit"); pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    row_is, _ = problem.A.getNestISs()
    pc.setFieldSplitIS(("u",row_is[0]),("rho",row_is[1]))
    pfx = ksp.getOptionsPrefix(); opts = PETSc.Options()
    opts[pfx+"fieldsplit_u_ksp_type"]   = "preonly"; opts[pfx+"fieldsplit_u_pc_type"]   = "hypre"
    opts[pfx+"fieldsplit_u_pc_hypre_type"] = "boomeramg"
    opts[pfx+"fieldsplit_rho_ksp_type"] = "preonly"; opts[pfx+"fieldsplit_rho_pc_type"] = "jacobi"
    problem.solver.setFromOptions()

    # D_pred^n = kappa * int |d-hat|^2 / rho-bar  (the dissipation term of the identity)
    D_rate_form = form(pde.k * ufl.inner(avg_domg, avg_domg) * 2/(rho+rho_prev) * dx)
    return pde, problem, (u, rho, u_prev, rho_prev), D_rate_form


def run_timeseries(K=32, dt=1e-3, T=0.5, every=10):
    """Figure 3 (left): normalised stored energy against time."""
    pde, problem, (u, rho, u_prev, rho_prev), _ = build(K, dt, "ts")
    comm = pde.spaces.mesh.comm
    E0 = pde.energy_total(u, rho)
    rows = [{"time": 0.0, "energy": E0, "energy_normalised": 1.0}]
    num_steps = int(round(T/dt))

    t = 0.0
    for step in range(num_steps):
        t += dt
        pde.assign_function(u_prev, u); pde.assign_function(rho_prev, rho)
        pde.assign_function(u, u_prev); pde.assign_function(rho, rho_prev)
        problem.solve()
        if problem.solver.getConvergedReason() <= 0:
            if comm.rank == 0: print(f"[timeseries] FAILED at step {step+1}")
            break
        if (step+1) % every == 0 or step+1 == num_steps:
            E = pde.energy_total(u, rho)
            rows.append({"time": t, "energy": E, "energy_normalised": E/E0})

    if comm.rank == 0:
        print(f"[timeseries] K={K} dt={dt} T={T}: "
              f"E(0)={E0:.6f}  E(T)/E(0)={rows[-1]['energy_normalised']*100:.2f}%")
        write_csv(os.path.join(BASE, "energy_vs_time.csv"),
                  ["time","energy","energy_normalised"], rows)
    return rows


def run_defect(K=64, T=0.5, dt_array=(0.05, 0.02, 0.01, 0.005)):
    """Figure 3 (right): cumulative energy-law defect against dt."""
    rows = []
    for dt in dt_array:
        pde, problem, (u, rho, u_prev, rho_prev), D_rate_form = build(K, dt, f"d{int(dt*1e4)}")
        comm = pde.spaces.mesh.comm
        E0 = pde.energy_total(u, rho)
        sum_D = 0.0
        for step in range(int(round(T/dt))):
            pde.assign_function(u_prev, u); pde.assign_function(rho_prev, rho)
            pde.assign_function(u, u_prev); pde.assign_function(rho, rho_prev)
            problem.solve()
            if problem.solver.getConvergedReason() <= 0:
                if comm.rank == 0: print(f"[defect dt={dt}] FAILED at step {step+1}")
                break
            sum_D += comm.allreduce(assemble_scalar(D_rate_form), op=MPI.SUM)
        defect = abs((pde.energy_total(u, rho) - E0) + dt*sum_D)
        if comm.rank == 0:
            print(f"[defect] dt={dt}  DEFECT={defect:.3e}")
        rows.append({"dt": dt, "defect": defect})

    if rows:
        print("\n--- DEFECT convergence (expect ~O(dt^2)) ---")
        for i, r in enumerate(rows):
            if i == 0:
                print(f"  dt={r['dt']:<7} DEFECT={r['defect']:.3e}")
            else:
                p = rows[i-1]
                rate = np.log(p["defect"]/r["defect"])/np.log(p["dt"]/r["dt"])
                print(f"  dt={r['dt']:<7} DEFECT={r['defect']:.3e}  ({rate:.2f})")
        write_csv(os.path.join(BASE, "defect_vs_dt.csv"), ["dt","defect"], rows)
    return rows


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("timeseries", "all"): run_timeseries()
    if what in ("defect", "all"):     run_defect()
    print(f"Done. {BASE}")
