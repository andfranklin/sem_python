from copy import deepcopy
import numpy as np

import os
import sys
base_dir = os.environ["SEM_PYTHON_DIR"]

sys.path.append(base_dir + "src/base")
from enums import ModelType, VariableName

sys.path.append(base_dir + "src/executioners")
from Executioner import Executioner, ExecutionerParameters

sys.path.append(base_dir + "src/solvers")
from NonlinearSolver import NonlinearSolver

class TransientExecutionerParameters(ExecutionerParameters):
  def __init__(self):
    ExecutionerParameters.__init__(self)
    self.registerFloatParameter("dt", "Nominal time step size")
    self.registerFloatParameter("end_time", "End time")

class TransientExecutioner(Executioner):
  def __init__(self, params, model, ics, bcs, eos_map, interface_closures, gravity, dof_handler, mesh, nonlinear_solver_params, factory):
    Executioner.__init__(self, params, model, ics, bcs, eos_map, interface_closures, gravity, dof_handler, mesh, nonlinear_solver_params, factory)
    self.dt_nominal = params.get("dt")
    self.end_time = params.get("end_time")
    self.U_old = deepcopy(self.U)

    self.M = self.computeMassMatrix()

  def computeMassMatrix(self):
    M = np.zeros(shape=(self.dof_handler.n_dof, self.dof_handler.n_dof))

    self.addMassMatrixPhase(M, 0)
    if (self.model_type != ModelType.OnePhase):
      self.addMassMatrixPhase(M, 1)
    if (self.model_type == ModelType.TwoPhase):
      self.addMassMatrixVolumeFraction(M)

    return M

  def addMassMatrixPhase(self, M, phase):
    phi = self.fe_values.get_phi()

    arho_index = self.dof_handler.variable_index[VariableName.ARho][phase]
    arhou_index = self.dof_handler.variable_index[VariableName.ARhoU][phase]
    arhoE_index = self.dof_handler.variable_index[VariableName.ARhoE][phase]

    for e in xrange(self.dof_handler.n_cell):
      M_cell = np.zeros(shape=(self.dof_handler.n_dof_per_cell, self.dof_handler.n_dof_per_cell))

      JxW = self.fe_values.get_JxW(e)
      for q in xrange(self.quadrature.n_q):
        for k_local in xrange(self.dof_handler.n_dof_per_cell_per_var):
          i_arho = self.dof_handler.i(k_local, arho_index)
          i_arhou = self.dof_handler.i(k_local, arhou_index)
          i_arhoE = self.dof_handler.i(k_local, arhoE_index)
          for l_local in xrange(self.dof_handler.n_dof_per_cell_per_var):
            j_arho = self.dof_handler.i(l_local, arho_index)
            j_arhou = self.dof_handler.i(l_local, arhou_index)
            j_arhoE = self.dof_handler.i(l_local, arhoE_index)

            M_cell[i_arho,j_arho] += phi[k_local,q] * phi[l_local,q] * JxW[q]
            M_cell[i_arhou,j_arhou] += phi[k_local,q] * phi[l_local,q] * JxW[q]
            M_cell[i_arhoE,j_arhoE] += phi[k_local,q] * phi[l_local,q] * JxW[q]

      # aggregate cell matrix into global matrix
      self.dof_handler.aggregateLocalMatrix(M, M_cell, e)

  def addMassMatrixVolumeFraction(self, M):
    phi = self.fe_values.get_phi()

    vf1_index = self.dof_handler.variable_index[VariableName.VF1][0]

    for e in xrange(self.dof_handler.n_cell):
      M_cell = np.zeros(shape=(self.dof_handler.n_dof_per_cell, self.dof_handler.n_dof_per_cell))

      JxW = self.fe_values.get_JxW(e)
      for q in xrange(self.quadrature.n_q):
        for k_local in xrange(self.dof_handler.n_dof_per_cell_per_var):
          i_vf1 = self.dof_handler.i(k_local, vf1_index)
          for l_local in xrange(self.dof_handler.n_dof_per_cell_per_var):
            j_vf1 = self.dof_handler.i(l_local, vf1_index)
            M_cell[i_vf1,j_vf1] += phi[k_local,q] * phi[l_local,q] * JxW[q]

      # aggregate cell matrix into global matrix
      self.dof_handler.aggregateLocalMatrix(M, M_cell, e)

  def assembleTransientSystem(self, U):
    M_dUdt = np.matmul(self.M, U - self.U_old)
    return (M_dUdt, self.M)

  def run(self):
    nonlinear_solver = NonlinearSolver(self.nonlinear_solver_params,
                                       self.assembleSystem,
                                       self.dof_handler)

    transient_incomplete = True
    t = 0.0
    time_step = 1
    while (transient_incomplete):
      # compute time step size
      if (t + self.dt_nominal >= self.end_time):
        transient_incomplete = False
        self.dt = self.end_time - t
      else:
        self.dt = self.dt_nominal
      t += self.dt

      print "\nTime step %i: t = %g, dt = %g" % (time_step, t, self.dt)

      nonlinear_solver.solve(self.U)
      self.U_old = deepcopy(self.U)

      time_step += 1

    return self.U
