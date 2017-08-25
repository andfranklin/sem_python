from enums import ModelType, VariableName
from BC import BC, BCParameters

class OnePhaseBCParameters(BCParameters):
  def __init__(self):
    BCParameters.__init__(self)
    self.registerIntParameter("phase", "Index of phase to which BC is applied")

class OnePhaseBC(BC):
  def __init__(self, params):
    BC.__init__(self, params)
    self.phase = params.get("phase")
    self.eos = self.eos_list[self.phase]

    # DoF indices for the conserved variables
    if (self.model_type == ModelType.TwoPhase):
      vf1_index = self.dof_handler.variable_index[VariableName.VF1][0]
      self.i_vf1 = self.dof_handler.i(self.k, vf1_index)

    arho_index = self.dof_handler.variable_index[VariableName.ARho][self.phase]
    arhou_index = self.dof_handler.variable_index[VariableName.ARhoU][self.phase]
    arhoE_index = self.dof_handler.variable_index[VariableName.ARhoE][self.phase]

    self.i_arho = self.dof_handler.i(self.k, arho_index)
    self.i_arhou = self.dof_handler.i(self.k, arhou_index)
    self.i_arhoE = self.dof_handler.i(self.k, arhoE_index)
