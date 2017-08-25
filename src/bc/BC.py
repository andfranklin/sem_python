from Parameters import Parameters

class BCParameters(Parameters):
  def __init__(self):
    Parameters.__init__(self)
    self.registerStringParameter("mesh_name", "Name of the mesh for which this BC applies")
    self.registerStringSelectionParameter("boundary", ["left", "right"], "Which boundary to apply boundary condition on")
    self.registerParameter("dof_handler", "Degree of freedom handler")
    self.registerParameter("eos", "Equation of state")

class BC(object):
  def __init__(self, params):
    self.mesh_name = params.get("mesh_name")
    self.boundary = params.get("boundary")
    self.dof_handler = params.get("dof_handler")
    self.eos = params.get("eos")

    self.model_type = self.dof_handler.model_type
    if self.boundary == "left":
      self.nx = -1.0
      self.k = self.dof_handler.getNodeIndexFromLeft(self.mesh_name, 0)
    else:
      self.nx = 1.0
      self.k = self.dof_handler.getNodeIndexFromRight(self.mesh_name, 0)

  def applyWeakBC(self, U, r, J):
    pass

  def applyStrongBCNonlinearSystem(self, U, r, J):
    pass

  def applyStrongBCLinearSystemMatrix(self, A):
    pass

  def applyStrongBCLinearSystemRHSVector(self, U_old, b):
    pass
