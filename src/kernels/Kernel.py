from abc import ABCMeta, abstractmethod
import numpy as np

import os
import sys
base_dir = os.environ["SEM_PYTHON_DIR"]

sys.path.append(base_dir + "src/input")
from Parameters import Parameters

class KernelParameters(Parameters):
  def __init__(self):
    Parameters.__init__(self)
    self.registerIntParameter("var_index", "Index of variable")
    self.registerBoolParameter("is_nodal", "Is this a nodal kernel?", False)

class Kernel(object):
  __metaclass__ = ABCMeta

  def __init__(self, params, dof_handler):
    self.var_index = params.get("var_index")
    self.is_nodal = params.get("is_nodal")
    self.dof_handler = dof_handler
    self.i = dof_handler.i
    if self.is_nodal:
      self.n = 1
    else:
      self.n = dof_handler.n_dof_per_cell_per_var
    self.zero = [0]

  ## Adds this kernel's contribution to the local residual vector and Jacobian
  #
  # @param[in] data  variables and aux quantities
  # @param[in] der  derivatives of variables and aux quantities
  # @param[inout] r  local residual vector
  # @param[inout] J  local Jacobian matrix
  def apply(self, data, der, r, J):
    # loop over test functions on element (loop over nodes)
    for i_local in xrange(self.n):
      i = self.i(i_local, self.var_index)
      r[i] += sum(self.computeResidual(data, i_local))
      # loop over basis functions on element (loop over nodes)
      for j_local in xrange(self.n):
        # loop over variables
        for var_index in self.var_indices:
          j = self.i(j_local, var_index)
          J[i,j] += sum(self.computeJacobian(data, der, var_index, i_local, j_local))

  ## Computes the kernel's residuals at the quadrature points when testing with
  #  local basis function \f$i\f$.
  #
  # @param[in] data  variables and aux quantities
  # @param[in] i  local test function index
  @abstractmethod
  def computeResidual(self, data, i):
    pass

  ## Computes the kernel's Jacobians at the quadrature points when testing with
  #  local basis function \f$i\f$, with respect to local basis function \f$j\f$.
  #
  # @param[in] data  variables and aux quantities
  # @param[in] der  derivatives of variables and aux quantities
  # @param[in] var  variable index
  # @param[in] i  local test function index
  # @param[in] j  local basis function index
  @abstractmethod
  def computeJacobian(self, data, der, var, i, j):
    pass
