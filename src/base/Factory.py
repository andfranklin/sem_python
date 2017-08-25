# aux
from AcousticImpedance import AcousticImpedance, AcousticImpedanceParameters
from AuxGradient import AuxGradient, AuxGradientParameters
from BerryInterfacePressure import BerryInterfacePressure, BerryInterfacePressureParameters
from BerryInterfacePressureBar import BerryInterfacePressureBar, BerryInterfacePressureBarParameters
from BerryInterfaceVelocity import BerryInterfaceVelocity, BerryInterfaceVelocityParameters
from BerryInterfaceVelocityBar import BerryInterfaceVelocityBar, BerryInterfaceVelocityBarParameters
from BerryInterfacialAreaDensity import BerryInterfacialAreaDensity, BerryInterfacialAreaDensityParameters
from BerryPressureRelaxationCoef import BerryPressureRelaxationCoef, BerryPressureRelaxationCoefParameters
from BerryVelocityRelaxationCoef import BerryVelocityRelaxationCoef, BerryVelocityRelaxationCoefParameters
from ConstantAux import ConstantAux, ConstantAuxParameters
from Density import Density, DensityParameters
from LaxFriedrichsCoefficient import LaxFriedrichsCoefficient, LaxFriedrichsCoefficientParameters
from LaxFriedrichsCoefficientVolumeFraction import LaxFriedrichsCoefficientVolumeFraction, LaxFriedrichsCoefficientVolumeFractionParameters
from EntropyMinimumMassFlux import EntropyMinimumMassFlux, EntropyMinimumMassFluxParameters
from EntropyMinimumMomentumFlux import EntropyMinimumMomentumFlux, EntropyMinimumMomentumFluxParameters
from EntropyMinimumEnergyFlux import EntropyMinimumEnergyFlux, EntropyMinimumEnergyFluxParameters
from EntropyMinimumVolumeFractionFlux import EntropyMinimumVolumeFractionFlux, EntropyMinimumVolumeFractionFluxParameters
from IdenticalAux import IdenticalAux, IdenticalAuxParameters
from InternalEnergyDensity import InternalEnergyDensity, InternalEnergyDensityParameters
from Pressure import Pressure, PressureParameters
from Temperature import Temperature, TemperatureParameters
from SoundSpeed import SoundSpeed, SoundSpeedParameters
from SpecificInternalEnergy import SpecificInternalEnergy, SpecificInternalEnergyParameters
from SpecificTotalEnergy import SpecificTotalEnergy, SpecificTotalEnergyParameters
from SpecificVolume import SpecificVolume, SpecificVolumeParameters
from Velocity import Velocity, VelocityParameters
from VolumeFraction1Phase import VolumeFraction1Phase, VolumeFraction1PhaseParameters
from VolumeFractionPhase1 import VolumeFractionPhase1, VolumeFractionPhase1Parameters
from VolumeFractionPhase2 import VolumeFractionPhase2, VolumeFractionPhase2Parameters
from AmbrosoBeta import AmbrosoBeta, AmbrosoBetaParameters
from AmbrosoVelocityRelaxationCoef import AmbrosoVelocityRelaxationCoef, AmbrosoVelocityRelaxationCoefParameters
from AmbrosoPressureRelaxationCoef import AmbrosoPressureRelaxationCoef, AmbrosoPressureRelaxationCoefParameters
from AmbrosoInterfaceVelocity import AmbrosoInterfaceVelocity, AmbrosoInterfaceVelocityParameters
from AmbrosoInterfacePressure import AmbrosoInterfacePressure, AmbrosoInterfacePressureParameters

# base
from Model import Model, ModelParameters

# bc
from FreeBC import FreeBC, FreeBCParameters
from DirichletVolumeFractionBC import DirichletVolumeFractionBC, DirichletVolumeFractionBCParameters
from InletRhoUBC import InletRhoUBC, InletRhoUBCParameters
from OutletBC import OutletBC, OutletBCParameters
from SolidWallBC import SolidWallBC, SolidWallBCParameters

# closures
from IdealGasEoS import IdealGasEoS, IdealGasEoSParameters
from StiffenedGasEoS import StiffenedGasEoS, StiffenedGasEoSParameters
from AmbrosoInterfaceClosures import AmbrosoInterfaceClosures, AmbrosoInterfaceClosuresParameters
from BerryInterfaceClosures import BerryInterfaceClosures, BerryInterfaceClosuresParameters
from ThermodynamicState import ThermodynamicState, ThermodynamicStateParameters

# executioners
from ExplicitEulerExecutioner import ExplicitEulerExecutioner, ExplicitEulerExecutionerParameters
from ImplicitEulerExecutioner import ImplicitEulerExecutioner, ImplicitEulerExecutionerParameters
from SteadyStateExecutioner import SteadyStateExecutioner, SteadyStateExecutionerParameters

# fem
from DoFHandler1Phase import DoFHandler1Phase, DoFHandler1PhaseParameters
from DoFHandler2PhaseNonInteracting import DoFHandler2PhaseNonInteracting, DoFHandler2PhaseNonInteractingParameters
from DoFHandler2Phase import DoFHandler2Phase, DoFHandler2PhaseParameters
from FEValues import FEValues, FEValuesParameters
from Quadrature import Quadrature, QuadratureParameters

# ic
from InitialConditions1Phase import InitialConditions1Phase, InitialConditions1PhaseParameters
from InitialConditions2Phase import InitialConditions2Phase, InitialConditions2PhaseParameters

# input
from PhysicsParameters import PhysicsParameters

# junctions
from CloneJunction import CloneJunction, CloneJunctionParameters
from CompressibleJunction import CompressibleJunction, CompressibleJunctionParameters

# kernels
from DissipationAuxFlux import DissipationAuxFlux, DissipationAuxFluxParameters
from DissipationVariableGradient import DissipationVariableGradient, DissipationVariableGradientParameters
from VolumeFractionAdvection import VolumeFractionAdvection, VolumeFractionAdvectionParameters
from VolumeFractionPressureRelaxation import VolumeFractionPressureRelaxation, VolumeFractionPressureRelaxationParameters
from MassAdvection import MassAdvection, MassAdvectionParameters
from MomentumAdvection import MomentumAdvection, MomentumAdvectionParameters
from MomentumGravity import MomentumGravity, MomentumGravityParameters
from MomentumVolumeFractionGradient import MomentumVolumeFractionGradient, MomentumVolumeFractionGradientParameters
from EnergyAdvection import EnergyAdvection, EnergyAdvectionParameters
from EnergyGravity import EnergyGravity, EnergyGravityParameters
from EnergyPressureRelaxation import EnergyPressureRelaxation, EnergyPressureRelaxationParameters
from EnergyVolumeFractionGradient import EnergyVolumeFractionGradient, EnergyVolumeFractionGradientParameters

# mesh
from UniformMesh import UniformMesh, UniformMeshParameters

# output
from Postprocessor import Postprocessor, PostprocessorParameters

# solvers
from NonlinearSolver import NonlinearSolver, NonlinearSolverParameters

# stabilization
from NoStabilization import NoStabilization, NoStabilizationParameters
from LaxFriedrichsStabilization import LaxFriedrichsStabilization, LaxFriedrichsStabilizationParameters

# utilities
from error_utilities import error

## Class for creating objects
class Factory(object):
  ## Creates a parameters object
  # @param object_class  class of object for which to create parameters object
  # @param params  dictionary of parameter names to their values as strings
  def createParametersObject(self, object_class, params):
    # parameters classes are always named as the object class plus "Parameters"
    parameters_class = object_class + "Parameters"

    # parameters classes should have no arguments to their constructors
    if parameters_class in globals():
      constructor = globals()[parameters_class]
    else:
      error("'" + parameters_class + "' is not a valid object type.")
    parameters_object = constructor()

    # set each of the parameters
    for param in params:
      if param != "type":
        parameters_object.set(param, params[param])

    return parameters_object

  ## Creates an object
  # @param object_class  class of object to create
  # @param params  dictionary of parameter names to their values as strings
  def createObject(self, object_class, params):
    # create the object's parameters object first
    parameters_object = self.createParametersObject(object_class, params)

    # create the object
    if object_class in globals():
      constructor = globals()[object_class]
    else:
      error("'" + object_class + "' is not a valid object type.")

    the_object = constructor(parameters_object)

    return the_object
