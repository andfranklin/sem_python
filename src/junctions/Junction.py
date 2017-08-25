from Parameters import Parameters
from error_utilities import error

class JunctionParameters(Parameters):
  def __init__(self):
    Parameters.__init__(self)
    self.registerStringListParameter("mesh_names", "List of names of the meshes to be connected")
    self.registerStringListParameter("mesh_sides", "List of sides of the meshes to be connected")
    self.registerParameter("dof_handler", "Degree of freedom handler")
    self.registerParameter("eos_list", "List of equations of state")

## Base class for a junction between meshes
class Junction(object):
  def __init__(self, params):
    self.mesh_names = params.get("mesh_names")
    self.mesh_sides = params.get("mesh_sides")
    self.dof_handler = params.get("dof_handler")
    self.eos_list = params.get("eos_list")

    self.model_type = self.dof_handler.model_type

    # ensure that lengths of list parameters agree
    self.n_meshes = len(self.mesh_names)
    if len(self.mesh_names) != len(self.mesh_sides):
      error("The list parameters 'mesh_names' and 'mesh_sides' must have the same size.")

    # get node indices and adjacent node indices
    self.node_indices = list()
    self.adjacent_node_indices = list()
    for i, mesh_name in enumerate(self.mesh_names):
      if self.mesh_sides[i] == "left":
        self.node_indices.append(self.dof_handler.getNodeIndexFromLeft(mesh_name, 0))
        self.adjacent_node_indices.append(self.dof_handler.getNodeIndexFromLeft(mesh_name, 1))
      elif self.mesh_sides[i] == "right":
        self.node_indices.append(self.dof_handler.getNodeIndexFromRight(mesh_name, 0))
        self.adjacent_node_indices.append(self.dof_handler.getNodeIndexFromRight(mesh_name, 1))
      else:
        error("Side parameters must be either 'left' or 'right'.")

  def applyWeaklyToNonlinearSystem(self, U, U_old, r, J):
    pass

  def applyStronglyToNonlinearSystem(self, U, U_old, r, J):
    pass

  def applyStronglyToLinearSystemMatrix(self, A):
    pass

  def applyStronglyToLinearSystemRHSVector(self, U_old, b):
    pass
