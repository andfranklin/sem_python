import os
import sys
base_dir = os.environ["SEM_PYTHON_DIR"]

sys.path.append(base_dir + "src/base")
from enums import ModelType

sys.path.append(base_dir + "src/input")
from Parameters import Parameters

class ModelParameters(Parameters):
  def __init__(self):
    Parameters.__init__(self)
    self.registerStringParameter("model", "Identifier for the model")
    self.registerBoolParameter("use_volume_fraction_pressure_relaxation",
                               "Use the pressure relaxation term in the volume fraction equation", True)
    self.registerBoolParameter("use_energy_pressure_relaxation",
                               "Use the pressure relaxation term in the energy equation", True)

class Model(object):
  def __init__(self, params):
    model_string_to_type = {"1phase": ModelType.OnePhase,
                            "2phase_noninteracting" : ModelType.TwoPhaseNonInteracting,
                            "2phase": ModelType.TwoPhase}
    self.model_type = model_string_to_type[params.get("model")]
    self.use_volume_fraction_pressure_relaxation = params.get("use_volume_fraction_pressure_relaxation")
    self.use_energy_pressure_relaxation = params.get("use_energy_pressure_relaxation")
