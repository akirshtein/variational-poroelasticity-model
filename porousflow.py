import os
import datetime
import shutil
import sys
import csv

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import logging
from logging import FileHandler
from types import SimpleNamespace

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, la, plot
from dolfinx.fem import (Constant, Function, Expression, dirichletbc,
                         extract_function_spaces, form, functionspace,
                         locate_dofs_topological,
                         assemble_scalar)
from dolfinx.fem.petsc import (
    LinearProblem,
    NonlinearProblem,
    apply_lifting,
    assemble_matrix, assemble_vector,
    create_vector,
    set_bc)
from dolfinx.io import VTKFile
from dolfinx.mesh import CellType,  create_rectangle, locate_entities_boundary, meshtags
from ufl import (div, dx, grad, inner, dot, 
                 TestFunction, TrialFunction, TestFunctions, TrialFunctions,
                 Identity, conditional, transpose, eq, ln, tanh)

sig=lambda x: (1+tanh(x))*(1-tanh(x))

class PorousFlow:
    def __init__(self, mu, lam, a, k, M, p=1, params=None):
        self.mu=mu
        self.lam=lam
        self.a=a
        self.k=k
        self.M=M
        self.p=p
        if p==1:
          self.o=1e-12
        elif p<1:
          print('cannot use p<1')
        self.params = params if params is not None else SimpleNamespace()
    def set_parameters(
        self,
        dt,
        K,
        T=1,
        formulation='threefield',
        ptype='fieldsplit',
        cell_type=CellType.quadrilateral,
        FrequencyOutput=1,
    ):
        self.params = SimpleNamespace()
        self.params.dt = dt
        self.params.K = K
        self.params.T = T
        self.params.formulation = formulation
        self.params.ptype = ptype
        self.params.cell_type = cell_type
        self.params.frequencyOutput = max(1, int(FrequencyOutput))
    def omega(self,rho):
      if self.p==1:
        return 0.5*self.M*rho*ln((rho**2)+self.o)
      elif self.p>1:
        return self.M*pow(rho,self.p)
    def domega(self,rho):
      if self.p==1:
        return 0.5*self.M*(ln((rho**2)+self.o)+2)
      elif self.p>1:
        return self.p*self.M*pow(rho,self.p-1)
    def pressure(self, rho):
      if self.p==1:
        return self.M*rho
      elif self.p>1:
        return self.M*(self.p-1)*pow(rho,self.p)
    def d2omega(self,rho):
      if self.p==1:
        return 0.5*self.M*pow((rho**2)+self.o, -0.5)
      elif self.p>1:
        return 0.5*self.p*(self.p-1)*self.M*pow(rho,self.p-2)
    def divomega(self, rhon, rhop):
        return conditional(eq(rhon,rhop),self.domega(rhop),(self.omega(rhon)-self.omega(rhop))/(rhon-rhop))
    def d1divomega(self, rhon, rhop):
        return conditional(eq(rhon,rhop),self.d2omega(rhop),(self.domega(rhon)-self.divomega(rhon,rhop))/(rhon-rhop))
    def sigma(self,u):
        return self.mu*(grad(u)+transpose(grad(u)))+self.lam*div(u)*Identity(2)
    def boundary(self, x):
        return np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)), 
                             np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
    def setup_spaces(self, build_boundary=False):
        self.spaces=SimpleNamespace()
        self.spaces.mesh = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [self.params.K, self.params.K], self.params.cell_type)
        self.spaces.all_boundary_facets = locate_entities_boundary(self.spaces.mesh, self.spaces.mesh.topology.dim - 1, self.boundary)
        cell = self.spaces.mesh.basix_cell()
        gdim = self.spaces.mesh.geometry.dim
        if self.params.formulation == 'threefield':
            Uel = element("Lagrange", cell, 2, shape=(gdim,))
            Qel = element("RT", cell, 1)
            Pel = element("DG", cell, 0)
            if self.params.ptype=='single':
                self.spaces.MSpace = functionspace(self.spaces.mesh, mixed_element([Uel, Qel, Pel]))
                # Locate boundary dofs for Dirichlet BCs
                self.spaces.U_boundary_dofs = locate_dofs_topological(self.spaces.MSpace.sub(0), self.spaces.mesh.topology.dim - 1, self.spaces.all_boundary_facets)
                self.spaces.Q_boundary_dofs = locate_dofs_topological(self.spaces.MSpace.sub(1), self.spaces.mesh.topology.dim - 1, self.spaces.all_boundary_facets)
            elif self.params.ptype=='fieldsplit':
                self.spaces.Uspace = functionspace(self.spaces.mesh, Uel)
                self.spaces.Qspace = functionspace(self.spaces.mesh, Qel)
                self.spaces.Pspace = functionspace(self.spaces.mesh, Pel)
                # Locate boundary dofs for Dirichlet BCs
                self.spaces.Uspace_boundary_dofs = locate_dofs_topological(self.spaces.Uspace, self.spaces.mesh.topology.dim - 1, self.spaces.all_boundary_facets)
                self.spaces.Qspace_boundary_dofs = locate_dofs_topological(self.spaces.Qspace, self.spaces.mesh.topology.dim - 1, self.spaces.all_boundary_facets)
            else:
                print('preconditioner type not recognized')
        elif self.params.formulation == 'twofield':
            Uel = element("Lagrange", cell, 2, shape=(gdim,))
            Pel = element("Lagrange", cell, 1)
            if self.params.ptype=='single':
                self.spaces.MSpace = functionspace(self.spaces.mesh, mixed_element([Uel, Pel]))
                # Locate boundary dofs for Dirichlet BCs
                self.spaces.U_boundary_dofs = locate_dofs_topological(self.spaces.MSpace.sub(0), self.spaces.mesh.topology.dim - 1, self.spaces.all_boundary_facets)
            elif self.params.ptype=='fieldsplit':
                self.spaces.Uspace = functionspace(self.spaces.mesh, Uel)
                self.spaces.Pspace = functionspace(self.spaces.mesh, Pel)
                # Locate boundary dofs for Dirichlet BCs
                self.spaces.Uspace_boundary_dofs = locate_dofs_topological(self.spaces.Uspace, self.spaces.mesh.topology.dim - 1, self.spaces.all_boundary_facets)
            else:
                print('preconditioner type not recognized')
        else:
            print('formulation not recognized')

        if build_boundary:
            self.build_boundary_facets()
    def test_and_trial_functions(self):
        if self.params.ptype=='single':
            dv=TrialFunction(self.spaces.MSpace)
            if self.params.formulation == 'threefield':
                (u, q, p) = ufl.split(dv)
                (v, w, r) = TestFunctions(self.spaces.MSpace)
                return dv, u, q, p, v, w, r
            elif self.params.formulation == 'twofield':
                (u, p) = ufl.split(dv)
                (v, q) = TestFunctions(self.spaces.MSpace)
                return dv, u, p, v, q
        elif self.params.ptype=='fieldsplit':
            if self.params.formulation == 'threefield':
                u = TrialFunction(self.spaces.Uspace)
                q = TrialFunction(self.spaces.Qspace)
                p = TrialFunction(self.spaces.Pspace)
                v = TestFunction(self.spaces.Uspace)
                w = TestFunction(self.spaces.Qspace)
                r = TestFunction(self.spaces.Pspace)
                return u, q, p, v, w, r
            elif self.params.formulation == 'twofield':
                u = TrialFunction(self.spaces.Uspace)
                p = TrialFunction(self.spaces.Pspace)
                v = TestFunction(self.spaces.Uspace)
                q = TestFunction(self.spaces.Pspace)
                return u, p, v, q

    def build_boundary_facets(self):
        """Build and store boundary facet index arrays for the unit-square sides."""
        mesh = self.spaces.mesh
        fdim = mesh.topology.dim - 1

        left = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0.0))
        right = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 1.0))
        bottom = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0.0))
        top = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 1.0))

        all_boundary = np.unique(np.concatenate([left, right, bottom, top]))
        self.spaces.facets = {
            "left": left,
            "right": right,
            "bottom": bottom,
            "top": top,
            "all": all_boundary,
        }
        return self.spaces.facets

    def build_ds(self, facets, marker=1):
        """Create a ds measure tagged on a supplied facet index array."""
        mesh = self.spaces.mesh
        values = np.full(facets.shape, marker, dtype=np.int32)
        facet_tag = meshtags(mesh, mesh.topology.dim - 1, facets, values)
        return ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

    def global_scalar(self, expr):
        """Assemble scalar and reduce across MPI ranks."""
        val_local = assemble_scalar(form(expr))
        return self.spaces.mesh.comm.allreduce(val_local, op=MPI.SUM)

    def assign_function(self, target, source):
        target.x.array[:] = source.x.array
        target.x.scatter_forward()

    def solve_initial_displacement(
        self,
        u,
        rho_like,
        displacement_bcs,
        traction=None,
        ds=None,
        petsc_options_prefix="pf_init_u_",
        petsc_options=None,
    ):
        trial_u = TrialFunction(self.spaces.Uspace)
        test_u = TestFunction(self.spaces.Uspace)

        a_form = inner(self.sigma(trial_u), grad(test_u)) * dx
        if self.params.formulation == "twofield":
            l_form = -dot(rho_like * grad(self.divomega(rho_like, rho_like)), self.a * test_u) * dx
        else:
            l_form = self.a * self.pressure(rho_like) * div(test_u) * dx
        if traction is not None and ds is not None:
            l_form += dot(traction, test_u) * ds(1)

        if petsc_options is None:
            petsc_options = {
                "ksp_type": "gmres",
                "ksp_rtol": 1e-10,
                "ksp_atol": 1e-12,
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
            }

        solution = LinearProblem(
            a_form,
            l_form,
            bcs=displacement_bcs,
            petsc_options_prefix=petsc_options_prefix,
            petsc_options=petsc_options,
        ).solve()
        self.assign_function(u, solution)

    def solve_initial_flux(
        self,
        q,
        rho_like,
        flux_bcs,
        petsc_options_prefix="pf_init_q_",
        petsc_options=None,
    ):
        trial_q = TrialFunction(self.spaces.Qspace)
        test_q = TestFunction(self.spaces.Qspace)

        a_form = inner(self.k**-1 * trial_q, test_q) * dx
        l_form = self.pressure(rho_like) * div(test_q) * dx

        if petsc_options is None:
            petsc_options = {
                "ksp_type": "gmres",
                "ksp_rtol": 1e-10,
                "ksp_atol": 1e-12,
                "pc_type": "jacobi",
            }

        solution = LinearProblem(
            a_form,
            l_form,
            bcs=flux_bcs,
            petsc_options_prefix=petsc_options_prefix,
            petsc_options=petsc_options,
        ).solve()
        self.assign_function(q, solution)

    def initialize_twofield_state(self, u, rho, u_prev, rho_prev, displacement_bcs, traction=None, ds=None):
        self.solve_initial_displacement(u, rho, displacement_bcs, traction=traction, ds=ds)
        self.assign_function(u_prev, u)
        self.assign_function(rho_prev, rho)

    def initialize_threefield_state(
        self,
        u,
        q,
        rho,
        u_prev,
        rho_prev,
        displacement_bcs,
        flux_bcs,
        traction=None,
        ds=None,
    ):
        self.solve_initial_displacement(u, rho, displacement_bcs, traction=traction, ds=ds)
        self.solve_initial_flux(q, rho, flux_bcs)
        self.assign_function(u_prev, u)
        self.assign_function(rho_prev, rho)

    def energy_total(self, u, rho_like):
        """Compute E = 0.5*(sigma(u), grad(u)) + (omega(rho), 1)."""
        elastic = 0.5 * self.global_scalar(ufl.inner(self.sigma(u), ufl.grad(u)) * ufl.dx)
        fluid = self.global_scalar(self.omega(rho_like) * ufl.dx)
        return elastic + fluid

    def mass_total(self, rho_like):
        return self.global_scalar(rho_like * ufl.dx)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)