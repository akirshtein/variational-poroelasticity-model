"""
Two-field footing driver — ramp-then-hold load schedule.  Usage:
    python PorousFlow_twofield_footing_driver.py <config_file>

Config (INI):
  [model]   mu, lam, alpha, kappa, M, gamma
  [numerics] K, dt, T, T_ramp, load
  [problem]  side_case  (clamped | sliding | free)

Load schedule:
  p_load(t) = load * t / T_ramp   for t in [0, T_ramp]
  p_load(t) = load                 for t in (T_ramp, T]

Reproduces the footing runs of Section 4.4 (Figure 5, Figures 6-8, Table 1) when run
with the nine configurations in configs/.

Writes, per configuration:
  history.csv      time series of load, stored energy, fluid mass, Newton iterations
  solution*.pvtu   VTK field snapshots (every ~num_steps/200 steps), used for the
                   deformed-domain figures

Output root defaults to ./output/footing/ next to this script; override with the
environment variable PF_OUTPUT_ROOT. NOTE: the VTK snapshots are large (~5 GB per
configuration at the paper's K=128, dt=1e-3, i.e. ~43 GB for all nine). Only
history.csv is needed for Figure 5 and Table 1.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from porousflow import *
import ufl, configparser

if len(sys.argv) != 2:
    raise SystemExit(f"Usage: python {sys.argv[0]} <config_file>")

cfg = configparser.ConfigParser()
cfg.read(sys.argv[1])

mu_val    = float(cfg["model"]["mu"])
lam_val   = float(cfg["model"]["lam"])
alpha_val = float(cfg["model"]["alpha"])
kappa_val = float(cfg["model"]["kappa"])
M_val     = float(cfg["model"]["M"])
gamma     = int(cfg["model"]["gamma"])

K        = int(cfg["numerics"]["K"])
dt       = float(cfg["numerics"]["dt"])
T        = float(cfg["numerics"]["T"])
T_ramp   = float(cfg["numerics"]["T_ramp"])
load_max = float(cfg["numerics"]["load"])

side_case = cfg["problem"]["side_case"].strip().lower()
if side_case not in ("clamped", "sliding", "free"):
    raise SystemExit(f"side_case must be clamped|sliding|free, got '{side_case}'")

num_steps = int(round(T / dt))
step_ramp = int(round(T_ramp / dt))
writing_frequency = max(1, int(round(num_steps / 200)))

output_root = os.environ.get(
    "PF_OUTPUT_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
outdir = os.path.join(output_root, "footing", f"{side_case}_g{gamma}")
ensure_dir(outdir)

pde = PorousFlow(mu=mu_val, lam=lam_val, a=alpha_val, k=kappa_val, M=M_val, p=gamma)
pde.set_parameters(dt=dt, K=K, T=T, formulation="twofield", ptype="fieldsplit",
                   FrequencyOutput=1, cell_type=CellType.triangle)
pde.setup_spaces(build_boundary=True)

u_trial, rho_trial, u_test, rho_test = pde.test_and_trial_functions()
u      = Function(pde.spaces.Uspace)
rho    = Function(pde.spaces.Pspace)
u_prev = Function(pde.spaces.Uspace)
rho_prev = Function(pde.spaces.Pspace)

rho.interpolate(lambda x: np.full(x.shape[1], 1.0))
rho_prev.interpolate(lambda x: np.full(x.shape[1], 1.0))

facets = pde.spaces.facets
ds_top = pde.build_ds(facets["top"], marker=1)
gdim   = pde.spaces.mesh.geometry.dim
zero_v = np.zeros(gdim, dtype=PETSc.ScalarType)
bcs    = []

bottom_dofs = locate_dofs_topological(
    pde.spaces.Uspace, pde.spaces.mesh.topology.dim-1, facets["bottom"])
bcs.append(dirichletbc(zero_v, bottom_dofs, pde.spaces.Uspace))

if side_case == "clamped":
    side_facets = np.unique(np.concatenate([facets["left"], facets["right"]]))
    bcs.append(dirichletbc(
        zero_v,
        locate_dofs_topological(pde.spaces.Uspace, pde.spaces.mesh.topology.dim-1, side_facets),
        pde.spaces.Uspace))
elif side_case == "sliding":
    side_facets = np.unique(np.concatenate([facets["left"], facets["right"]]))
    bcs.append(dirichletbc(
        PETSc.ScalarType(0.0),
        locate_dofs_topological(pde.spaces.Uspace.sub(0), pde.spaces.mesh.topology.dim-1, side_facets),
        pde.spaces.Uspace.sub(0)))

dt_c = Constant(pde.spaces.mesh, PETSc.ScalarType(dt))
half = Constant(pde.spaces.mesh, PETSc.ScalarType(0.5))
#stab = Constant(pde.spaces.mesh, PETSc.ScalarType(1.0))
stab=dt_c
load = Constant(pde.spaces.mesh, PETSc.ScalarType(0.0))
g    = ufl.as_vector((PETSc.ScalarType(0.0), -load))

pde.initialize_twofield_state(u, rho, u_prev, rho_prev, list(bcs), traction=g, ds=ds_top)

top_rho_dofs = locate_dofs_topological(
    pde.spaces.Pspace, pde.spaces.mesh.topology.dim-1, facets["top"])
bcs.append(dirichletbc(PETSc.ScalarType(1.0), top_rho_dofs, pde.spaces.Pspace))

avg_rho     = half*(rho + rho_prev)
avg_domg    = (avg_rho*grad(pde.domega(rho_prev))
               + rho_prev*grad(pde.d2omega(rho_prev)*(rho - rho_prev)))

R_u   = (half*inner(pde.sigma((1+stab)*u+(1-stab)*u_prev), grad(u_test))*dx
         + dot(avg_domg, pde.a*u_test)*dx
         - dot(g, u_test)*ds_top(1))
R_rho = (dot(avg_domg, dt_c*pde.k*grad(rho_test))*dx
         + dot(-pde.a*avg_rho*(u-u_prev), grad(rho_test))*dx
         + (rho-rho_prev)*rho_test*dx)

J11=ufl.derivative(R_u,  u,   u_trial); J12=ufl.derivative(R_u,  rho, rho_trial)
J21=ufl.derivative(R_rho,u,   u_trial); J22=ufl.derivative(R_rho,rho, rho_trial)

problem = NonlinearProblem(
    [R_u, R_rho], [u, rho], bcs=bcs,
    J=[[J11,J12],[J21,J22]], P=[[J11,None],[None,J22]], kind="nest",
    petsc_options_prefix=f"pf2lo_{side_case}_",
    petsc_options={"snes_type":"newtonls","snes_rtol":1e-6,"snes_atol":1e-7,
                   "snes_max_it":60,"snes_converged_reason":None,
                   "ksp_type":"gmres","ksp_rtol":1e-5,"ksp_atol":1e-7,
                   "pc_type":"fieldsplit"},
)
ksp=problem.solver.getKSP(); pc=ksp.getPC()
pc.setType("fieldsplit"); pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
row_is,_=problem.A.getNestISs()
pc.setFieldSplitIS(("u",row_is[0]),("rho",row_is[1]))
pfx=ksp.getOptionsPrefix(); opts=PETSc.Options()
opts[f"{pfx}fieldsplit_u_ksp_type"]  ="preonly"
opts[f"{pfx}fieldsplit_u_pc_type"]   ="hypre"
opts[f"{pfx}fieldsplit_u_pc_hypre_type"]="boomeramg"
opts[f"{pfx}fieldsplit_rho_ksp_type"]="preonly"
opts[f"{pfx}fieldsplit_rho_pc_type"] ="jacobi"
problem.solver.setFromOptions()

D_pred_form = form(pde.k * ufl.inner(avg_domg, avg_domg) * 2/(rho+rho_prev) * dx)
W_ext_form  = form(dot(g, u-u_prev)*ds_top(1))

comm    = pde.spaces.mesh.comm
u_out   = Function(pde.spaces.Uspace); u_out.name  = "u"
q_out   = Function(pde.spaces.Uspace); q_out.name  = "q"
rho_out = Function(pde.spaces.Pspace); rho_out.name= "rho"
p_out   = Function(pde.spaces.Pspace); p_out.name  = "p"
ut_out  = Function(pde.spaces.Uspace); ut_out.name = "u_t"

def write_vtk(vtk, t_val):
    u_out.interpolate(u)
    q_out.interpolate(Expression(-pde.k*grad(pde.pressure(rho)),
                                 pde.spaces.Uspace.element.interpolation_points))
    rho_out.interpolate(rho)
    p_out.interpolate(Expression(pde.pressure(rho),
                                 pde.spaces.Pspace.element.interpolation_points))
    ut_out.x.array[:] = (u.x.array - u_prev.x.array)/dt
    ut_out.x.scatter_forward()
    vtk.write_function(u_out, t_val); vtk.write_function(q_out,   t_val)
    vtk.write_function(rho_out,t_val); vtk.write_function(p_out,   t_val)
    vtk.write_function(ut_out, t_val)

t = 0.0; W_ext_cumul = 0.0; history = []
E0 = pde.energy_total(u, rho); M0 = pde.mass_total(rho)
history.append({"step":0,"time":0.0,"phase":"init","load":0.0,
                "energy":E0,"delta_energy":0.0,"mass":M0,"mass_error":0.0,
                "W_ext_step":0.0,"W_ext_cumul":0.0,
                "D_pred":0.0,"D_inferred":0.0,"snes_it":0,"converged_reason":0})

if comm.rank==0:
    print(f"[{side_case}_g{gamma}] T_ramp={T_ramp} T={T} step_ramp={step_ramp} num_steps={num_steps}", flush=True)

with VTKFile(comm, os.path.join(outdir,"solution.pvd"), "w") as vtk:
    write_vtk(vtk, 0.0)

    for step in range(1, num_steps+1):
        t += dt
        if step <= step_ramp:
            load.value = PETSc.ScalarType(load_max * step / step_ramp)
            phase = "ramp"
        else:
            load.value = PETSc.ScalarType(load_max)
            phase = "hold"

        pde.assign_function(u_prev, u); pde.assign_function(rho_prev, rho)
        pde.assign_function(u, u_prev); pde.assign_function(rho, rho_prev)

        problem.solve()
        reason  = problem.solver.getConvergedReason()
        snes_it = problem.solver.getIterationNumber()
        if reason<=0 and comm.rank==0:
            print(f"  FAIL step={step} reason={reason}", flush=True)

        E_prev     = history[-1]["energy"]
        E          = pde.energy_total(u, rho)
        M          = pde.mass_total(rho)
        W_ext_step = comm.allreduce(assemble_scalar(W_ext_form), op=MPI.SUM)
        D_pred     = comm.allreduce(assemble_scalar(D_pred_form),op=MPI.SUM)
        W_ext_cumul += W_ext_step
        D_inferred  = -(E - E_prev - W_ext_step)/dt

        history.append({"step":step,"time":t,"phase":phase,"load":float(load.value),
                        "energy":E,"delta_energy":E-E_prev,"mass":M,"mass_error":M-M0,
                        "W_ext_step":W_ext_step,"W_ext_cumul":W_ext_cumul,
                        "D_pred":D_pred,"D_inferred":D_inferred,
                        "snes_it":snes_it,"converged_reason":reason})

        if step % writing_frequency == 0:
            write_vtk(vtk, t)
            if comm.rank==0:
                print(f"  VTK @ step={step} t={t:.3f} [{phase}] load={float(load.value):.4f}", flush=True)

        if comm.rank==0 and step%200==0:
            print(f"  step={step}/{num_steps} t={t:.3f} [{phase}]"
                  f" E={E:.6f} M-M0={M-M0:.3e}", flush=True)

fieldnames=["step","time","phase","load","energy","delta_energy","mass","mass_error",
            "W_ext_step","W_ext_cumul","D_pred","D_inferred","snes_it","converged_reason"]
write_csv(os.path.join(outdir,"history.csv"), fieldnames, history)
if comm.rank==0:
    print(f"Done: {outdir}", flush=True)
