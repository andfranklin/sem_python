from Junction import Junction, JunctionParameters
from error_utilities import error

class CloneJunctionParameters(JunctionParameters):
  def __init__(self):
    JunctionParameters.__init__(self)

## Junction that clones the master node solution values to the slave node solution values
class CloneJunction(Junction):
  def __init__(self, params):
    Junction.__init__(self, params)
    if self.n_meshes != 2:
      error("CloneJunction is only implemented for connecting 2 meshes.")

    # assume that first mesh is the "master"
    self.k_master = self.node_indices[0]
    self.k_slave = self.node_indices[1]

    # create lists of corresponding DoF indices
    self.variable_indices = range(self.dof_handler.n_var)
    self.i_master = [self.dof_handler.i(self.k_master, m) for m in self.variable_indices]
    self.i_slave = [self.dof_handler.i(self.k_slave, m) for m in self.variable_indices]

  def applyWeaklyToNonlinearSystem(self, U, U_old, r, J):
    pass

  def applyStronglyToNonlinearSystem(self, U, U_old, r, J):
    # add slave node residuals and Jacobians to master node
    r[self.i_master] += r[self.i_slave]
    r[self.i_slave] = 0
    J[self.i_master,:] += J[self.i_slave,:]
    J[self.i_slave,:] = 0
    J[:,self.i_master] += J[:,self.i_slave]
    J[:,self.i_slave] = 0

    # apply Dirichlet constraint to slave nodes
    r[self.i_slave] = U[self.i_slave] - U[self.i_master]
    J[self.i_slave,self.i_slave] = 1
    J[self.i_slave,self.i_master] = -1

  def applyStronglyToLinearSystemMatrix(self, A):
    A[self.i_slave,:] = 0
    A[self.i_slave,self.i_slave] = 1
    A[self.i_slave,self.i_master] = -1

  def applyStronglyToLinearSystemRHSVector(self, U_old, b):
    b[self.i_slave] = 0
