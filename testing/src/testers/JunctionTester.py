from copy import deepcopy
import numpy as np

from display_utilities import computeRelativeDifferenceMatrix, printRelativeMatrixDifference
from enums import ModelType
from Factory import Factory
from numeric_utilities import computeRelativeDifference

class JunctionTester(object):
  def __init__(self, junction_name, verbose=False):
    self.junction_name = junction_name
    self.verbose = verbose

  def checkJacobian(self, test_weak, model_type=ModelType.OnePhase, phase=0, junction_params=dict(), fd_eps=1e-8):
    # factory
    factory = Factory()

    # meshes
    params1 = {"n_cell": 1, "name": "mesh1"}
    params2 = {"n_cell": 1, "name": "mesh2"}
    meshes = [factory.createObject("UniformMesh", params1), factory.createObject("UniformMesh", params2)]

    # DoF handler
    dof_handler_params = {"meshes": meshes}
    if model_type == ModelType.OnePhase:
      dof_handler_class = "DoFHandler1Phase"
    elif model_type == ModelType.TwoPhaseNonInteracting:
      dof_handler_class = "DoFHandler2PhaseNonInteracting"
      def vf1_initial(x):
        return 0.3
      dof_handler_params["initial_vf1"] = vf1_initial
    elif model_type == ModelType.TwoPhase:
      dof_handler_class = "DoFHandler2Phase"
    dof_handler = factory.createObject(dof_handler_class, dof_handler_params)
    n_dof = dof_handler.n_dof
    n_var = dof_handler.n_var

    # junction
    junction_params["mesh_names"] = " ".join([mesh.name for mesh in meshes])
    junction_params["mesh_sides"] = "right left"
    junction_params["dof_handler"] = dof_handler
    junction = factory.createObject(self.junction_name, junction_params)

    # compute base solution
    U = np.zeros(n_dof)
    for i in xrange(n_dof):
      U[i] = i + 1.0

    # determine evaluation function
    if test_weak:
      f = junction.applyWeaklyToNonlinearSystem
    else:
      f = junction.applyStronglyToNonlinearSystem

    # base calculation
    r = np.zeros(n_dof)
    J_hand_coded = np.zeros(shape=(n_dof, n_dof))
    f(U, r, J_hand_coded)

    # finite difference Jacobians
    rel_diffs = np.zeros(shape=(n_dof, n_dof))
    J_fd = np.zeros(shape=(n_dof, n_dof))
    for j in xrange(n_dof):
      # perturb solution
      U_perturbed = deepcopy(U)
      U_perturbed[j] += fd_eps

      # compute finite difference Jacobian
      r_perturbed = np.zeros(n_dof)
      J_perturbed = np.zeros(shape=(n_dof, n_dof))
      f(U_perturbed, r_perturbed, J_perturbed)
      for i in xrange(n_dof):
        J_fd[i,j] = (r_perturbed[i] - r[i]) / fd_eps

    # compute relative difference matrix
    rel_diffs = computeRelativeDifferenceMatrix(J_hand_coded, J_fd)

    # print results
    if self.verbose:
      if test_weak:
        print "Jacobian of weak contributions:"
      else:
        print "Jacobian of strong contributions:"
      printRelativeMatrixDifference(rel_diffs, J_hand_coded - J_fd, 1e-1, 1e-3)

    # take the absolute value of the relative differences
    return abs(rel_diffs)
