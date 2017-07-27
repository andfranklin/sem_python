import numpy as np

import os
import sys
base_dir = os.environ["SEM_PYTHON_DIR"]

sys.path.append(base_dir + "src/base")
from enums import ModelType, VariableName

sys.path.append(base_dir + "src/closures")
from thermodynamic_functions import computeVolumeFraction, computeVelocity, \
  computeDensity, computeSpecificVolume, computeSpecificTotalEnergy, \
  computeSpecificInternalEnergy

sys.path.append(base_dir + "src/fem")
from FEValues import FEValues
from Quadrature import Quadrature

sys.path.append(base_dir + "src/input")
from Parameters import Parameters

class ExecutionerParameters(Parameters):
  def __init__(self):
    Parameters.__init__(self)
    self.registerBoolParameter("split_source", "Use source-splitting?", False)

class Executioner(object):
  def __init__(self, params, model_type, ics, bcs, eos, interface_closures, gravity, dof_handler, mesh, nonlinear_solver_params, stabilization, factory):
    self.split_source = params.get("split_source")

    self.model_type = model_type
    self.bcs = bcs
    self.eos = eos
    self.gravity = gravity
    self.dof_handler = dof_handler
    self.mesh = mesh
    self.nonlinear_solver_params = nonlinear_solver_params
    self.quadrature = Quadrature()
    self.fe_values = FEValues(self.quadrature, dof_handler, mesh)
    self.factory = factory
    self.need_solution_gradients = stabilization.needSolutionGradients()

    # initialize the solution
    self.U = np.zeros(self.dof_handler.n_dof)
    self.initializePhaseSolution(ics, 0)
    if self.model_type != ModelType.OnePhase:
      self.initializePhaseSolution(ics, 1)
    if self.model_type == ModelType.TwoPhase:
      self.initializeVolumeFractionSolution(ics)

    # set local solution update function
    if self.model_type == ModelType.OnePhase:
      self.computeLocalCellSolution = self.computeLocalCellSolutionOnePhase
      self.computeLocalNodeSolution = self.computeLocalNodeSolutionOnePhase
    else:
      self.computeLocalCellSolution = self.computeLocalCellSolutionTwoPhase
      self.computeLocalNodeSolution = self.computeLocalNodeSolutionTwoPhase

    # create aux quantities
    self.aux_list = self.createIndependentPhaseAuxQuantities(0) \
      + stabilization.createIndependentPhaseAuxQuantities(0)
    if self.model_type != ModelType.OnePhase:
      self.aux_list += self.createIndependentPhaseAuxQuantities(1) \
        + stabilization.createIndependentPhaseAuxQuantities(1)
    if self.model_type == ModelType.TwoPhase:
      self.aux_list += interface_closures.createAuxQuantities() \
        + stabilization.createPhaseInteractionAuxQuantities()

    # create list of source kernels
    self.source_kernels = self.createIndependentPhaseSourceKernels(0)
    if self.model_type != ModelType.OnePhase:
      self.source_kernels += self.createIndependentPhaseSourceKernels(1)
    if self.model_type == ModelType.TwoPhase:
      self.source_kernels += self.createPhaseInteractionSourceKernels()

    # create advection kernels
    self.fem_kernels = self.createIndependentPhaseAdvectionKernels(0) + stabilization.createIndependentPhaseKernels(0)
    if self.model_type != ModelType.OnePhase:
      self.fem_kernels += self.createIndependentPhaseAdvectionKernels(1) + stabilization.createIndependentPhaseKernels(1)
    if self.model_type == ModelType.TwoPhase:
      self.fem_kernels += self.createPhaseInteractionAdvectionKernels() + stabilization.createPhaseInteractionKernels()

    # add source kernels to kernel lists if not using source-splitting
    if not self.split_source:
      self.fem_kernels += self.source_kernels

  def initializePhaseSolution(self, ics, phase):
    # get appropriate volume fraction function
    if self.model_type == ModelType.OnePhase:
      def initial_vf(x):
        return 1
    else:
      initial_vf1 = ics.vf1
      if phase == 0:
        initial_vf = initial_vf1
      else:
        def initial_vf(x):
          return 1 - initial_vf1(x)

    # get relevant IC functions
    initial_p = ics.p[phase]
    initial_u = ics.u[phase]
    if ics.specified_rho:
      initial_rho = ics.rho[phase]
    else:
      initial_T = ics.T[phase]

    # compute IC
    eos_phase = self.eos[phase]
    arho_index = self.dof_handler.variable_index[VariableName.ARho][phase]
    arhou_index = self.dof_handler.variable_index[VariableName.ARhoU][phase]
    arhoE_index = self.dof_handler.variable_index[VariableName.ARhoE][phase]
    for k in xrange(self.dof_handler.n_node):
      vf = initial_vf(self.mesh.x[k])
      p = initial_p(self.mesh.x[k])
      u = initial_u(self.mesh.x[k])
      if ics.specified_rho:
        rho = initial_rho(self.mesh.x[k])
      else:
        T = initial_T(self.mesh.x[k])
        rho = eos_phase.rho(p, T)
      e = eos_phase.e(1.0 / rho, p)[0]
      E = e + 0.5 * u * u
      self.U[self.dof_handler.i(k, arho_index)] = vf * rho
      self.U[self.dof_handler.i(k, arhou_index)] = vf * rho * u
      self.U[self.dof_handler.i(k, arhoE_index)] = vf * rho * E

  def initializeVolumeFractionSolution(self, ics):
    vf1_index = self.dof_handler.variable_index[VariableName.VF1][0]
    for k in xrange(self.dof_handler.n_node):
      self.U[self.dof_handler.i(k, vf1_index)] = ics.vf1(self.mesh.x[k])

  def createIndependentPhaseAuxQuantities(self, phase):
    # create list of aux quantities to create
    aux_names_phase = list()
    if phase == 0:
      if self.model_type == ModelType.OnePhase:
        aux_names_phase.append("VolumeFraction1Phase")
      else:
        aux_names_phase.append("VolumeFractionPhase1")
    else:
      aux_names_phase.append("VolumeFractionPhase2")
    aux_names_phase += ["Velocity", "SpecificTotalEnergy", "Density", \
      "SpecificVolume", "SpecificInternalEnergy", "Pressure", "Temperature", "SoundSpeed"]

    # create the aux quantities for this phase
    aux_list = list()
    for aux_name in aux_names_phase:
      params = {"phase": phase}
      if aux_name == "Pressure":
        params["p_function"] = self.eos[phase].p
      elif aux_name == "Temperature":
        params["T_function"] = self.eos[phase].T
      elif aux_name == "SoundSpeed":
        params["c_function"] = self.eos[phase].c
      aux_list.append(self.factory.createObject(aux_name, params))

    return aux_list

  def createIndependentPhaseAdvectionKernels(self, phase):
    params = dict()
    params["phase"] = phase
    args = tuple([self.dof_handler])

    kernel_names = ["MassAdvection", "MomentumAdvection", "EnergyAdvection"]
    kernels = [self.factory.createObject(kernel_name, params, args) for kernel_name in kernel_names]

    return kernels

  def createIndependentPhaseSourceKernels(self, phase):
    params = dict()
    params["phase"] = phase
    if self.split_source:
      params["is_nodal"] = True
    args = tuple([self.dof_handler])

    kernel_names = ["MomentumGravity", "EnergyGravity"]
    kernels = [self.factory.createObject(kernel_name, params, args) for kernel_name in kernel_names]

    return kernels

  def createPhaseInteractionAdvectionKernels(self):
    params1 = dict()
    params2 = dict()
    params1["phase"] = 0
    params2["phase"] = 1
    args = tuple([self.dof_handler])

    kernels = list()
    kernels.append(self.factory.createObject("VolumeFractionAdvection", params1, args))
    kernels.append(self.factory.createObject("MomentumVolumeFractionGradient", params1, args))
    kernels.append(self.factory.createObject("MomentumVolumeFractionGradient", params2, args))
    kernels.append(self.factory.createObject("EnergyVolumeFractionGradient", params1, args))
    kernels.append(self.factory.createObject("EnergyVolumeFractionGradient", params2, args))

    return kernels

  def createPhaseInteractionSourceKernels(self):
    params1 = dict()
    params2 = dict()
    params1["phase"] = 0
    params2["phase"] = 1
    if self.split_source:
      params1["is_nodal"] = True
      params2["is_nodal"] = True
    args = tuple([self.dof_handler])

    kernels = list()
    kernels.append(self.factory.createObject("VolumeFractionPressureRelaxation", params1, args))
    kernels.append(self.factory.createObject("EnergyPressureRelaxation", params1, args))
    kernels.append(self.factory.createObject("EnergyPressureRelaxation", params2, args))

    return kernels

  # computes the steady-state residual and Jacobian without applying strong BC
  def assembleSteadyStateSystemWithoutStrongBC(self, U):
    r = np.zeros(self.dof_handler.n_dof)
    J = np.zeros(shape=(self.dof_handler.n_dof, self.dof_handler.n_dof))

    # volumetric terms
    self.addSteadyStateSystem(U, r, J)

    # weak boundary terms
    for bc in self.bcs:
      bc.applyWeakBC(U, r, J)

    return (r, J)

  # computes the full steady-state residual and Jacobian (strong BC applied)
  def assembleSteadyStateSystem(self, U):
    r, J = self.assembleSteadyStateSystemWithoutStrongBC(U)
    self.applyStrongBCNonlinearSystem(U, r, J)
    return (r, J)

  ## Applies strong BC to a nonlinear system solved with Newton's method
  # @param[in] U  implicit solution vector
  # @param[in] r  nonlinear system residual vector
  # @param[in] J  nonlinear system Jacobian matrix
  def applyStrongBCNonlinearSystem(self, U, r, J):
    for bc in self.bcs:
      bc.applyStrongBCNonlinearSystem(U, r, J)

  ## Applies strong BC to a linear system matrix.
  #
  # This is separated from the corresponding RHS vector modification function
  # because the matrix needs to be modified only once; the RHS vector might
  # depend on time or the solution vector.
  #
  # @param[in] A      linear system matrix
  def applyStrongBCLinearSystemMatrix(self, A):
    for bc in self.bcs:
      bc.applyStrongBCLinearSystemMatrix(A)

  ## Applies strong BC to a linear system RHS vector.
  # @param[in] U_old  old solution, needed if Dirichlet values are solution-dependent
  # @param[in] b      linear system RHS vector
  def applyStrongBCLinearSystemRHSVector(self, U_old, b):
    for bc in self.bcs:
      bc.applyStrongBCLinearSystemRHSVector(U_old, b)

  ## Computes the steady-state residual and Jacobian
  def addSteadyStateSystem(self, U, r, J):
    data = dict()
    der = dict()

    data["phi"] = self.fe_values.get_phi()
    data["g"] = self.gravity

    for elem in xrange(self.dof_handler.n_cell):
      r_cell = np.zeros(self.dof_handler.n_dof_per_cell)
      J_cell = np.zeros(shape=(self.dof_handler.n_dof_per_cell, self.dof_handler.n_dof_per_cell))

      data["grad_phi"] = self.fe_values.get_grad_phi(elem)
      data["JxW"] = self.fe_values.get_JxW(elem)
      data["dx"] = self.mesh.h[elem]

      # compute solution
      self.computeLocalCellSolution(U, elem, data)

      # compute auxiliary quantities
      for aux in self.aux_list:
        aux.compute(data, der)

      # compute the local residual and Jacobian
      for kernel in self.fem_kernels:
        kernel.apply(data, der, r_cell, J_cell)

      # aggregate cell residual and matrix into global residual and matrix
      self.dof_handler.aggregateLocalCellVector(r, r_cell, elem)
      self.dof_handler.aggregateLocalCellMatrix(J, J_cell, elem)

  ## Computes the local cell solution and gradients for 1-phase flow
  def computeLocalCellSolutionOnePhase(self, U, elem, data):
    data["vf1"] = self.fe_values.computeLocalVolumeFractionSolution(U, elem)
    data["arho1"] = self.fe_values.computeLocalSolution(U, VariableName.ARho, 0, elem)
    data["arhou1"] = self.fe_values.computeLocalSolution(U, VariableName.ARhoU, 0, elem)
    data["arhoE1"] = self.fe_values.computeLocalSolution(U, VariableName.ARhoE, 0, elem)
    if self.need_solution_gradients:
      data["grad_vf1"] = self.fe_values.computeLocalVolumeFractionSolutionGradient(U, elem)
      data["grad_arho1"] = self.fe_values.computeLocalSolutionGradient(U, VariableName.ARho, 0, elem)
      data["grad_arhou1"] = self.fe_values.computeLocalSolutionGradient(U, VariableName.ARhoU, 0, elem)
      data["grad_arhoE1"] = self.fe_values.computeLocalSolutionGradient(U, VariableName.ARhoE, 0, elem)

  ## Computes the local node solution for 1-phase flow
  def computeLocalNodeSolutionOnePhase(self, U, k, data):
    arho1_index = self.dof_handler.arho_index[0]
    arhou1_index = self.dof_handler.arhou_index[0]
    arhoE1_index = self.dof_handler.arhoE_index[0]
    data["vf1"] = self.dof_handler.getVolumeFraction(U, k)
    data["arho1"] = U[self.dof_handler.i(k, arho1_index)]
    data["arhou1"] = U[self.dof_handler.i(k, arhou1_index)]
    data["arhoE1"] = U[self.dof_handler.i(k, arhoE1_index)]

  ## Computes the local cell solution and gradients for 2-phase flow
  def computeLocalCellSolutionTwoPhase(self, U, elem, data):
    data["vf1"] = self.fe_values.computeLocalVolumeFractionSolution(U, elem)
    data["arho1"] = self.fe_values.computeLocalSolution(U, VariableName.ARho, 0, elem)
    data["arhou1"] = self.fe_values.computeLocalSolution(U, VariableName.ARhoU, 0, elem)
    data["arhoE1"] = self.fe_values.computeLocalSolution(U, VariableName.ARhoE, 0, elem)
    data["arho2"] = self.fe_values.computeLocalSolution(U, VariableName.ARho, 1, elem)
    data["arhou2"] = self.fe_values.computeLocalSolution(U, VariableName.ARhoU, 1, elem)
    data["arhoE2"] = self.fe_values.computeLocalSolution(U, VariableName.ARhoE, 1, elem)
    data["grad_vf1"] = self.fe_values.computeLocalVolumeFractionSolutionGradient(U, elem)
    if self.need_solution_gradients:
      data["grad_arho1"] = self.fe_values.computeLocalSolutionGradient(U, VariableName.ARho, 0, elem)
      data["grad_arhou1"] = self.fe_values.computeLocalSolutionGradient(U, VariableName.ARhoU, 0, elem)
      data["grad_arhoE1"] = self.fe_values.computeLocalSolutionGradient(U, VariableName.ARhoE, 0, elem)
      data["grad_arho2"] = self.fe_values.computeLocalSolutionGradient(U, VariableName.ARho, 1, elem)
      data["grad_arhou2"] = self.fe_values.computeLocalSolutionGradient(U, VariableName.ARhoU, 1, elem)
      data["grad_arhoE2"] = self.fe_values.computeLocalSolutionGradient(U, VariableName.ARhoE, 1, elem)

  ## Computes the local node solution for 2-phase flow
  def computeLocalNodeSolutionTwoPhase(self, U, k, data):
    arho1_index = self.dof_handler.arho_index[0]
    arhou1_index = self.dof_handler.arhou_index[0]
    arhoE1_index = self.dof_handler.arhoE_index[0]
    arho2_index = self.dof_handler.arho_index[1]
    arhou2_index = self.dof_handler.arhou_index[1]
    arhoE2_index = self.dof_handler.arhoE_index[1]
    data["vf1"] = self.dof_handler.getVolumeFraction(U, k)
    data["arho1"] = U[self.dof_handler.i(k, arho1_index)]
    data["arhou1"] = U[self.dof_handler.i(k, arhou1_index)]
    data["arhoE1"] = U[self.dof_handler.i(k, arhoE1_index)]
    data["arho2"] = U[self.dof_handler.i(k, arho2_index)]
    data["arhou2"] = U[self.dof_handler.i(k, arhou2_index)]
    data["arhoE2"] = U[self.dof_handler.i(k, arhoE2_index)]

  def solve(self):
    self.nonlinear_solver.solve(self.U)
