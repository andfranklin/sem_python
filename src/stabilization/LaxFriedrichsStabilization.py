from enums import VariableName, ModelType
from Stabilization import Stabilization, StabilizationParameters

class LaxFriedrichsStabilizationParameters(StabilizationParameters):
  def __init__(self):
    StabilizationParameters.__init__(self)
    self.registerFloatParameter("mult", "Multiplier for the viscous coefficient", 1.0)
    self.registerBoolParameter("use_simple_dissipation", "Use simple form of dissipation", False)

## Lax-Friedrichs stabilization
class LaxFriedrichsStabilization(Stabilization):
  def __init__(self, params):
    Stabilization.__init__(self, params)
    self.mult = params.get("mult")
    self.use_simple_dissipation = params.get("use_simple_dissipation")

  def needSolutionGradients(self):
    return True

  def createIndependentPhaseAuxQuantities(self, phase):
    aux_list = list()
    var_list = ["arho", "arhou", "arhoE"]

    # add the viscous coefficients
    for var in var_list:
      params = {"phase": phase, "var": var, "mult": self.mult}
      aux_list.append(self.factory.createObject("LaxFriedrichsCoefficient", params))

    if not self.use_simple_dissipation:
      # internal energy density
      params = {"phase": phase}
      aux_list.append(self.factory.createObject("InternalEnergyDensity", params))

      # gradients
      aux_gradient_names = ["vf", "rho", "u", "rhoe"]
      if phase == 0:
        variable_names = ["vf1", "arho1", "arhou1", "arhoE1"]
      else:
        variable_names = ["vf1", "arho2", "arhou2", "arhoE2"]
      for aux_gradient_name in aux_gradient_names:
        params = {"aux": aux_gradient_name + str(phase + 1), "variable_names": variable_names}
        aux_list.append(self.factory.createObject("AuxGradient", params))

      # add the viscous fluxes
      flux_classes = ["EntropyMinimumVolumeFractionFlux", "EntropyMinimumMassFlux",
        "EntropyMinimumMomentumFlux", "EntropyMinimumEnergyFlux"]
      params = {"phase": phase}
      for flux_class in flux_classes:
        aux_list.append(self.factory.createObject(flux_class, params))

    return aux_list

  def createAuxQuantities(self):
    aux_list = list()

    if self.model_type == ModelType.TwoPhase:
      params = {"mult": self.mult}
      aux_list.append(self.factory.createObject("LaxFriedrichsCoefficientVolumeFraction", params))
    else:
      # create zero coefficient for volume fraction equation
      params = {"name": "visccoef_vf", "value": 0}
      aux_list.append(self.factory.createObject("ConstantAux", params))

    aux_list += self.createIndependentPhaseAuxQuantities(0)
    if self.model_type != ModelType.OnePhase:
      aux_list += self.createIndependentPhaseAuxQuantities(1)

    return aux_list

  def createIndependentPhaseKernels(self, phase):
    kernels = list()
    var_enums = [VariableName.ARho, VariableName.ARhoU, VariableName.ARhoE]
    for var_enum in var_enums:
      if self.use_simple_dissipation:
        params = {"phase": phase, "dof_handler": self.dof_handler, "var_enum": var_enum}
        kernels.append(self.factory.createObject("DissipationVariableGradient", params))
      else:
        var_name = self.dof_handler.variableEnumToName(var_enum, phase)
        flux_name = "viscflux_" + var_name
        params = {"phase": phase, "dof_handler": self.dof_handler, "var_enum": var_enum, "flux_name": flux_name}
        kernels.append(self.factory.createObject("DissipationAuxFlux", params))
    return kernels

  def createPhaseInteractionKernels(self):
    kernels = list()
    if self.use_simple_dissipation:
      params = {"phase": 0, "dof_handler": self.dof_handler, "var_enum": VariableName.VF1}
      kernels.append(self.factory.createObject("DissipationVariableGradient", params))
    else:
      params = {"phase": 0, "dof_handler": self.dof_handler, "var_enum": VariableName.VF1, "flux_name": "viscflux_vf1"}
      kernels.append(self.factory.createObject("DissipationAuxFlux", params))
    return kernels
