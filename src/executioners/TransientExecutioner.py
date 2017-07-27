from copy import deepcopy
import numpy as np
from termcolor import colored

import os
import sys
base_dir = os.environ["SEM_PYTHON_DIR"]

sys.path.append(base_dir + "src/base")
from enums import ModelType, VariableName

sys.path.append(base_dir + "src/executioners")
from Executioner import Executioner, ExecutionerParameters

sys.path.append(base_dir + "src/utilities")
from error_utilities import error

class TransientExecutionerParameters(ExecutionerParameters):
  def __init__(self):
    ExecutionerParameters.__init__(self)
    self.registerFloatParameter("dt", "Nominal time step size")
    self.registerFloatParameter("cfl", "CFL number to compute time step size")
    self.registerFloatParameter("end_time", "End time")
    self.registerBoolParameter("lump_mass_matrix", "Lump the mass matrix?", False)

class TransientExecutioner(Executioner):
  def __init__(self, params, model_type, ics, bcs, eos_map, interface_closures, gravity, dof_handler, mesh, nonlinear_solver_params, stabilization, factory):
    Executioner.__init__(self, params, model_type, ics, bcs, eos_map, interface_closures, gravity, dof_handler, mesh, nonlinear_solver_params, stabilization, factory)

    # determine how time step size is determined
    if params.has("dt") and params.has("cfl"):
      error("The parameters 'dt' and 'cfl' cannot both be provided.")
    elif params.has("dt"):
      self.dt_nominal = params.get("dt")
      self.use_cfl_dt = False
    elif params.has("cfl"):
      self.cfl = params.get("cfl")
      self.use_cfl_dt = True
      self.cfl_dt_aux_list = self.createCFLAuxQuantities()
    else:
      error("Either parameter 'dt' or 'cfl' must be provided.")

    self.end_time = params.get("end_time")
    self.lump_mass_matrix = params.get("lump_mass_matrix")

    # tolerance to prevent small final time steps due to floating point precision error
    self.end_tolerance = 1e-12

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
            if self.lump_mass_matrix:
              M_cell[i_arho,i_arho] += phi[k_local,q] * phi[l_local,q] * JxW[q]
              M_cell[i_arhou,i_arhou] += phi[k_local,q] * phi[l_local,q] * JxW[q]
              M_cell[i_arhoE,i_arhoE] += phi[k_local,q] * phi[l_local,q] * JxW[q]
            else:
              j_arho = self.dof_handler.i(l_local, arho_index)
              j_arhou = self.dof_handler.i(l_local, arhou_index)
              j_arhoE = self.dof_handler.i(l_local, arhoE_index)

              M_cell[i_arho,j_arho] += phi[k_local,q] * phi[l_local,q] * JxW[q]
              M_cell[i_arhou,j_arhou] += phi[k_local,q] * phi[l_local,q] * JxW[q]
              M_cell[i_arhoE,j_arhoE] += phi[k_local,q] * phi[l_local,q] * JxW[q]

      # aggregate cell matrix into global matrix
      self.dof_handler.aggregateLocalCellMatrix(M, M_cell, e)

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
            if self.lump_mass_matrix:
              M_cell[i_vf1,i_vf1] += phi[k_local,q] * phi[l_local,q] * JxW[q]
            else:
              j_vf1 = self.dof_handler.i(l_local, vf1_index)
              M_cell[i_vf1,j_vf1] += phi[k_local,q] * phi[l_local,q] * JxW[q]

      # aggregate cell matrix into global matrix
      self.dof_handler.aggregateLocalCellMatrix(M, M_cell, e)

  def assembleTransientSystem(self, U):
    M_dU = np.matmul(self.M, U - self.U_old)
    return (M_dU, self.M)

  ## Integrates the source terms when source-splitting is used
  def takeSourceStep(self, U, dt):
    data = dict()
    der = dict()

    data["phi"] = np.zeros(shape=(self.dof_handler.n_dof_per_cell_per_var, 1))
    data["grad_phi"] = np.zeros(shape=(self.dof_handler.n_dof_per_cell_per_var, 1))
    data["phi"].fill(1.0)
    data["grad_phi"].fill(float("NaN"))
    data["JxW"] = 1.0
    data["g"] = self.gravity

    n_var = self.dof_handler.n_var

    # loop over nodes
    for k in xrange(self.dof_handler.n_node):
      r_node = np.zeros(n_var)
      J_node = np.zeros(shape=(n_var, n_var))

      # compute solution
      self.computeLocalNodeSolution(U, k, data)

      # compute auxiliary quantities
      for aux in self.aux_list:
        aux.compute(data, der)

      # compute the local residual and Jacobian
      for kernel in self.source_kernels:
        kernel.apply(data, der, r_node, J_node)

      # add integrated source (recall kernels assume LHS)
      self.dof_handler.aggregateLocalNodeVector(U, -dt * r_node, k)

  def createCFLAuxQuantities(self):
    aux_list = list()

    if self.model_type == ModelType.OnePhase:
      vf_classes = ["VolumeFraction1Phase"]
      self.n_phases = 1
    else:
      vf_classes = ["VolumeFractionPhase1", "VolumeFractionPhase2"]
      self.n_phases = 2

    for phase in xrange(self.n_phases):
      names = [vf_classes[phase]] + ["Density", "SpecificVolume", "Velocity", "SpecificTotalEnergy",
        "SpecificInternalEnergy", "Pressure", "SoundSpeed"]
      for name in names:
        if name == "Pressure":
          params = {"phase": phase, "p_function": self.eos[phase].p}
        elif name == "SoundSpeed":
          params = {"phase": phase, "c_function": self.eos[phase].c}
        else:
          params = {"phase": phase}
        aux_list.append(self.factory.createObject(name, params))

    return aux_list

  def computeTimeStepSizeFromCFL(self):
    # determine minimum cell width
    dx_min = self.mesh.getMinimumCellWidth()

    # determine maximum wave speed
    data = dict()
    der = dict()

    for phase in xrange(self.n_phases):
      phase_str = str(phase + 1)
      vf = "vf" + phase_str
      arho = "arho" + phase_str
      arhou = "arhou" + phase_str
      arhoE = "arhoE" + phase_str
      data[vf], data[arho], data[arhou], data[arhoE] = self.dof_handler.getPhaseSolution(self.U_old, phase)

    for aux in self.cfl_dt_aux_list:
      aux.compute(data, der)

    wave_speed_max = 0
    for phase in xrange(self.n_phases):
      phase_str = str(phase + 1)
      u = "u" + phase_str
      c = "c" + phase_str
      wave_speed = "wave_speed" + phase_str

      data[wave_speed] = abs(data[u]) + data[c]
      wave_speed_max = max(wave_speed_max, max(data[wave_speed]))

    # compute CFL time step size
    dt_CFL = dx_min / wave_speed_max

    return self.cfl * dt_CFL

  def run(self):
    transient_incomplete = True
    t = 0.0
    time_step = 1
    print ""
    while (transient_incomplete):
      # compute time step size
      if self.use_cfl_dt:
        dt = self.computeTimeStepSizeFromCFL()
      else:
        dt = self.dt_nominal

      # shorten time step if at end of transient
      if (t + dt + self.end_tolerance >= self.end_time):
        transient_incomplete = False
        self.dt = self.end_time - t
      else:
        self.dt = dt

      # update time
      t += self.dt

      print "Time step %i: t = %g, dt = %g" % (time_step, t, self.dt)

      try:
        # solve the time step
        self.solve()
      except:
        # report failure and exit transient
        print colored("Time step failed.", "red")
        return self.U_old

      # perform source term integration
      if self.split_source:
        self.takeSourceStep(self.U, self.dt)

      # store the old solution
      self.U_old = deepcopy(self.U)

      # save old solution and increment time step index
      self.U_old = deepcopy(self.U)
      time_step += 1

    return self.U

  def solve(self):
    if self.dt > 1e-15:
      residual_factor = self.dt
    else:
      residual_factor = 1
    self.nonlinear_solver.solve(self.U, residual_factor=residual_factor)
