import os
import sys
base_dir = os.environ["SEM_PYTHON_DIR"]

sys.path.append(base_dir + "src/aux")
from AuxQuantity1Phase import AuxQuantity1Phase, AuxQuantity1PhaseParameters

class TemperatureParameters(AuxQuantity1PhaseParameters):
  def __init__(self):
    AuxQuantity1PhaseParameters.__init__(self)
    self.registerFunctionParameter("T_function", "Temperature function from EOS")

class Temperature(AuxQuantity1Phase):
  def __init__(self, params):
    AuxQuantity1Phase.__init__(self, params)
    self.T_function = params.get("T_function")

  def compute(self, data, der):
    T, dT_dv, dT_de = self.T_function(data[self.v], data[self.e])
    data[self.T] = T

    dT_dvf1 = dT_dv * der[self.v]["vf1"]
    dT_darho = dT_dv * der[self.v][self.arho] + dT_de * der[self.e][self.arho]
    dT_darhou = dT_de * der[self.e][self.arhou]
    dT_darhoE = dT_de * der[self.e][self.arhoE]
    der[self.T] = {"vf1" : dT_dvf1, self.arho : dT_darho, self.arhou : dT_darhou, self.arhoE : dT_darhoE}