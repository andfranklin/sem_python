import os
import sys
base_dir = os.environ["SEM_PYTHON_DIR"]

sys.path.append(base_dir + "src/executioners")
from Executioner import Executioner, ExecutionerParameters

sys.path.append(base_dir + "src/solvers")
from NonlinearSolver import NonlinearSolver

class SteadyStateExecutionerParameters(ExecutionerParameters):
  def __init__(self):
    ExecutionerParameters.__init__(self)

class SteadyStateExecutioner(Executioner):
  def __init__(self, params, model, ics, bcs, eos_map, interface_closures, gravity, dof_handler, mesh, nonlinear_solver_params, factory):
    Executioner.__init__(self, params, model, ics, bcs, eos_map, interface_closures, gravity, dof_handler, mesh, nonlinear_solver_params, factory)

  def run(self):
    nonlinear_solver = NonlinearSolver(self.nonlinear_solver_params,
      self.assembleSteadyStateSystem, self.dof_handler)
    nonlinear_solver.solve(self.U)

    return self.U
