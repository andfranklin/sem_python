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

class Executioner(object):
  def __init__(self, params, model, ics, bcs, eos, interface_closures, gravity, dof_handler, mesh, nonlinear_solver_params, factory):
    self.model = model
    self.model_type = self.model.type
    self.bcs = bcs
    self.eos = eos
    self.interface_closures = interface_closures
    self.gravity = gravity
    self.dof_handler = dof_handler
    self.mesh = mesh
    self.nonlinear_solver_params = nonlinear_solver_params
    self.quadrature = Quadrature()
    self.fe_values = FEValues(self.quadrature, dof_handler, mesh)
    self.factory = factory

    # initialize the solution
    self.U = np.zeros(self.dof_handler.n_dof)
    self.initializePhaseSolution(ics, 0)
    if self.model_type != ModelType.OnePhase:
      self.initializePhaseSolution(ics, 1)
    if self.model_type == ModelType.TwoPhase:
      self.initializeVolumeFractionSolution(ics)

    # create aux quantities
    self.aux1 = self.createIndependentPhaseAuxQuantities(0)
    if self.model_type != ModelType.OnePhase:
      self.aux2 = self.createIndependentPhaseAuxQuantities(1)
    if self.model_type == ModelType.TwoPhase:
      self.aux_2phase = self.aux1 + self.aux2 + self.createPhaseInteractionAuxQuantities()

    # create kernels
    self.kernels1 = self.createIndependentPhaseKernels(0)
    if self.model_type != ModelType.OnePhase:
      self.kernels2 = self.createIndependentPhaseKernels(1)
    if self.model_type == ModelType.TwoPhase:
      self.kernels_2phase = self.createPhaseInteractionKernels()

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
                           "SpecificVolume", "SpecificInternalEnergy", "Pressure", "Temperature"]

    # create the aux quantities for this phase
    aux_list = list()
    for aux_name in aux_names_phase:
      params = {"phase": phase}
      if aux_name == "Pressure":
        params["p_function"] = self.eos[phase].p
      elif aux_name == "Temperature":
        params["T_function"] = self.eos[phase].T
      aux_list.append(self.factory.createObject(aux_name, params))

    return aux_list

  def createPhaseInteractionAuxQuantities(self):
    interaction_aux_names = ["Beta", "Mu", "Theta", "InterfaceVelocity", "InterfacePressure"]
    interaction_aux = list()
    for aux_name in interaction_aux_names:
      params = dict()
      if aux_name == "Beta":
        params["beta_function"] = self.interface_closures.computeBeta
      elif aux_name == "Mu":
        params["mu_function"] = self.interface_closures.computeMu
      elif aux_name == "Theta":
        params["theta_function"] = self.interface_closures.computeTheta
      elif aux_name == "InterfaceVelocity":
        params["uI_function"] = self.interface_closures.computeInterfaceVelocity
      elif aux_name == "InterfacePressure":
        params["pI_function"] = self.interface_closures.computeInterfacePressure
      interaction_aux.append(self.factory.createObject(aux_name, params))

    return interaction_aux

  def createIndependentPhaseKernels(self, phase):
    kernels = list()
    params = dict()
    params["phase"] = phase
    args = tuple([self.dof_handler])
    kernel_name_list = ["MassAdvection", "MomentumAdvection", "MomentumGravity", "EnergyAdvection", "EnergyGravity"]
    for kernel_name in kernel_name_list:
      kernels.append(self.factory.createObject(kernel_name, params, args))
    return kernels

  def createPhaseInteractionKernels(self):
    kernels = list()

    params1 = dict()
    params2 = dict()
    params1["phase"] = 0
    params2["phase"] = 1
    args = tuple([self.dof_handler])

    kernels.append(self.factory.createObject("VolumeFractionAdvection", params1, args))
    kernels.append(self.factory.createObject("VolumeFractionPressureRelaxation", params1, args))
    kernels.append(self.factory.createObject("MomentumVolumeFractionGradient", params1, args))
    kernels.append(self.factory.createObject("MomentumVolumeFractionGradient", params2, args))
    kernels.append(self.factory.createObject("EnergyPressureRelaxation", params1, args))
    kernels.append(self.factory.createObject("EnergyPressureRelaxation", params2, args))
    kernels.append(self.factory.createObject("EnergyVolumeFractionGradient", params1, args))
    kernels.append(self.factory.createObject("EnergyVolumeFractionGradient", params2, args))

    return kernels

  # computes the steady-state residual and Jacobian without applying strong BC
  def assembleSteadyStateSystemWithoutStrongBC(self, U):
    r = np.zeros(self.dof_handler.n_dof)
    J = np.zeros(shape=(self.dof_handler.n_dof, self.dof_handler.n_dof))

    # volumetric terms
    self.addSteadyStateSystemPhase(U, self.kernels1, self.aux1, 0, r, J)
    if (self.model_type != ModelType.OnePhase):
      self.addSteadyStateSystemPhase(U, self.kernels2, self.aux2, 1, r, J)
    if (self.model_type == ModelType.TwoPhase):
      self.addSteadyStateSystemVolumeFraction(U, r, J)

    # weak boundary terms
    for bc in self.bcs:
      bc.applyWeakBC(U, r, J)

    return (r, J)

  # computes the full steady-state residual and Jacobian (strong BC applied)
  def assembleSteadyStateSystem(self, U):
    r, J = self.assembleSteadyStateSystemWithoutStrongBC(U)
    self.applyStrongBC(U, r, J)
    return (r, J)

  # applies strong BC
  def applyStrongBC(self, U, r, J):
    for bc in self.bcs:
      bc.applyStrongBC(U, r, J)

  # computes the steady-state residual and Jacobian for a phase
  def addSteadyStateSystemPhase(self, U, kernel_list, aux_list, phase, r, J):
    data = dict()
    der = dict()

    data["phi"] = self.fe_values.get_phi()
    data["g"] = self.gravity

    arho_name = "arho" + str(phase + 1)
    arhou_name = "arhou" + str(phase + 1)
    arhoE_name = "arhoE" + str(phase + 1)

    for elem in xrange(self.dof_handler.n_cell):
      r_cell = np.zeros(self.dof_handler.n_dof_per_cell)
      J_cell = np.zeros(shape=(self.dof_handler.n_dof_per_cell, self.dof_handler.n_dof_per_cell))

      data["grad_phi"] = self.fe_values.get_grad_phi(elem)
      data["JxW"] = self.fe_values.get_JxW(elem)

      # compute solution
      data["vf1"] = self.fe_values.computeLocalVolumeFractionSolution(U, elem)
      data[arho_name] = self.fe_values.computeLocalSolution(U, VariableName.ARho, phase, elem)
      data[arhou_name] = self.fe_values.computeLocalSolution(U, VariableName.ARhoU, phase, elem)
      data[arhoE_name] = self.fe_values.computeLocalSolution(U, VariableName.ARhoE, phase, elem)

      # compute auxiliary quantities
      for aux in aux_list:
        aux.compute(data, der)

      # compute the local residual and Jacobian
      for kernel in kernel_list:
        kernel.apply(data, der, r_cell, J_cell)

      # aggregate cell residual and matrix into global residual and matrix
      self.dof_handler.aggregateLocalVector(r, r_cell, elem)
      self.dof_handler.aggregateLocalMatrix(J, J_cell, elem)

  # computes the steady-state residual for the volume fraction equation
  def addSteadyStateSystemVolumeFraction(self, U, r, J):
    data = dict()
    der = dict()

    data["phi"] = self.fe_values.get_phi()
    data["g"] = self.gravity

    for elem in xrange(self.dof_handler.n_cell):
      r_cell = np.zeros(self.dof_handler.n_dof_per_cell)
      J_cell = np.zeros(shape=(self.dof_handler.n_dof_per_cell, self.dof_handler.n_dof_per_cell))

      data["grad_phi"] = self.fe_values.get_grad_phi(elem)
      data["JxW"] = self.fe_values.get_JxW(elem)

      # compute solution
      data["vf1"] = self.fe_values.computeLocalSolution(U, VariableName.VF1, 0, elem)
      data["arho1"] = self.fe_values.computeLocalSolution(U, VariableName.ARho, 0, elem)
      data["arhou1"] = self.fe_values.computeLocalSolution(U, VariableName.ARhoU, 0, elem)
      data["arhoE1"] = self.fe_values.computeLocalSolution(U, VariableName.ARhoE, 0, elem)
      data["arho2"] = self.fe_values.computeLocalSolution(U, VariableName.ARho, 1, elem)
      data["arhou2"] = self.fe_values.computeLocalSolution(U, VariableName.ARhoU, 1, elem)
      data["arhoE2"] = self.fe_values.computeLocalSolution(U, VariableName.ARhoE, 1, elem)

      # compute solution gradient
      data["dvf1_dx"] = self.fe_values.computeLocalSolutionGradient(U, VariableName.VF1, 0, elem)

      # compute auxiliary quantities
      for aux in self.aux_2phase:
        aux.compute(data, der)

      # compute the local residual and Jacobian
      for kernel in self.kernels_2phase:
        kernel.apply(data, der, r_cell, J_cell)

      # aggregate cell residual and matrix into global residual and matrix
      self.dof_handler.aggregateLocalVector(r, r_cell, elem)
      self.dof_handler.aggregateLocalMatrix(J, J_cell, elem)
