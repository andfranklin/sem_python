from abc import ABCMeta, abstractmethod

import os
import sys
base_dir = os.environ["SEM_PYTHON_DIR"]

sys.path.append(base_dir + "src/input")
from Parameters import Parameters

class InterfaceClosuresParameters(Parameters):
  def __init__(self):
    Parameters.__init__(self)
    self.registerParameter("factory", "Factory")

class InterfaceClosures(object):
  __metaclass__ = ABCMeta

  def __init__(self, params):
    self.factory = params.get("factory")

  @abstractmethod
  def createAuxQuantities(self):
    pass
