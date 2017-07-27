from abc import ABCMeta, abstractmethod
import numpy as np

import os
import sys
base_dir = os.environ["SEM_PYTHON_DIR"]

sys.path.append(base_dir + "src/base")
from enums import ModelType, VariableName

sys.path.append(base_dir + "src/closures")
from thermodynamic_functions import computeVolumeFraction

class DoFHandler(object):
  __metaclass__ = ABCMeta
  def __init__(self, mesh):
    # counts
    self.n_cell = mesh.n_cell
    self.n_dof_per_var = self.n_cell + 1
    self.n_node = self.n_dof_per_var
    self.n_dof_per_cell_per_var = 2

  def setup(self):
    if (self.model_type == ModelType.OnePhase):
      n_phases = 1
      n_vf_equations = 0
    elif (self.model_type == ModelType.TwoPhaseNonInteracting):
      n_phases = 2
      n_vf_equations = 0
    elif (self.model_type == ModelType.TwoPhase):
      n_phases = 2
      n_vf_equations = 1
    else:
      raise NotImplementedError("Selected model type not implemented")

    arho_index_phase = 0
    arhou_index_phase = 2
    arhoE_index_phase = 1
    self.n_var = n_vf_equations + n_phases * 3
    self.arho_index = list()
    self.arhou_index = list()
    self.arhoE_index = list()
    self.variable_names = dict()
    self.index_to_variable = dict()
    self.index_to_phase = dict()
    for phase in xrange(n_phases):
      self.arho_index.append(n_vf_equations + phase * 3 + arho_index_phase)
      self.arhou_index.append(n_vf_equations + phase * 3 + arhou_index_phase)
      self.arhoE_index.append(n_vf_equations + phase * 3 + arhoE_index_phase)

      self.variable_names[self.arho_index[phase]] = "arho" + str(phase+1)
      self.variable_names[self.arhou_index[phase]] = "arhou" + str(phase+1)
      self.variable_names[self.arhoE_index[phase]] = "arhoE" + str(phase+1)

      self.index_to_variable[self.arho_index[phase]] = VariableName.ARho
      self.index_to_variable[self.arhou_index[phase]] = VariableName.ARhoU
      self.index_to_variable[self.arhoE_index[phase]] = VariableName.ARhoE

      self.index_to_phase[self.arho_index[phase]] = phase
      self.index_to_phase[self.arhou_index[phase]] = phase
      self.index_to_phase[self.arhoE_index[phase]] = phase

    self.variable_index = {
      VariableName.ARho: self.arho_index,
      VariableName.ARhoU: self.arhou_index,
      VariableName.ARhoE: self.arhoE_index}

    if self.model_type == ModelType.TwoPhase:
      self.vf1_index = [0]
      phase = 0
      self.variable_index[VariableName.VF1] = self.vf1_index
      self.variable_names[self.vf1_index[phase]] = "vf1"
      self.index_to_variable[self.vf1_index[phase]] = VariableName.VF1
      self.index_to_phase[self.vf1_index[phase]] = phase

    # total number of DoFs
    self.n_dof_per_cell = self.n_dof_per_cell_per_var * self.n_var
    self.n_dof = self.n_dof_per_var * self.n_var

  # DoF index
  def i(self, k, var_index):
    return k * self.n_var + var_index

  # global node index
  def k(self, e, k_local):
    return e + k_local

  ## Converts variable enum to its string name with phase index
  # @param[in] var  variable enum
  # @param[in] phase  phase
  def variableEnumToName(self, var, phase):
    index = self.variable_index[var][phase]
    return self.variable_names[index]

  @abstractmethod
  def getVolumeFraction(self, U, k):
    pass

  def getSolution(self, U, variable_name, phase):
    var_index = self.variable_index[variable_name][phase]
    return np.array([U[self.i(k, var_index)] for k in xrange(self.n_node)])

  def getPhaseSolution(self, U, phase):
    vf1 = np.array([self.getVolumeFraction(U, k) for k in xrange(self.n_node)])
    vf, _ = computeVolumeFraction(vf1, phase, self.model_type)
    arho = self.getSolution(U, VariableName.ARho, phase)
    arhou = self.getSolution(U, VariableName.ARhoU, phase)
    arhoE = self.getSolution(U, VariableName.ARhoE, phase)
    return (vf, arho, arhou, arhoE)

  # aggregates local cell vector into global vector
  def aggregateLocalCellVector(self, r, r_cell, e):
    r[e * self.n_var : (e+2) * self.n_var] += r_cell

  # aggregates local node vector into global vector
  def aggregateLocalNodeVector(self, r, r_node, k):
    r[k * self.n_var : (k+1) * self.n_var] += r_node

  # aggregates local cell matrix into global matrix
  def aggregateLocalCellMatrix(self, J, J_cell, e):
    J[e * self.n_var : (e+2) * self.n_var, e * self.n_var : (e+2) * self.n_var] += J_cell

  # aggregates local node matrix into global matrix
  def aggregateLocalNodeMatrix(self, J, J_node, k):
    J[k * self.n_var : (k+1) * self.n_var, k * self.n_var : (k+1) * self.n_var] += J_node

  ## Applies scaling factors to the nonlinear residual based on variable
  # @param[in,out] r  nonlinear residual vector to modify
  # @param[in] scaling  dictionary of variable name and phase to scaling factor
  def applyScalingFactors(self, r, scaling):
    for m in xrange(self.n_var):
      variable_m = self.index_to_variable[m]
      phase_m = self.index_to_phase[m]
      scaling_m = scaling[variable_m][phase_m]
      for k in xrange(self.n_node):
        r[self.i(k, m)] *= scaling_m
