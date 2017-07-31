from abc import ABCMeta, abstractmethod

import os
import sys
base_dir = os.environ["SEM_PYTHON_DIR"]

sys.path.append(base_dir + "src/input")
from Parameters import Parameters

class EoSParameters(Parameters):
  def __init__(self):
    Parameters.__init__(self)

## Abstract base class for equations of state
class EoS(object):
  __metaclass__ = ABCMeta
  @abstractmethod
  def rho(self, p, T):
    pass

  @abstractmethod
  def e(self, v, p):
    pass

  @abstractmethod
  def p(self, v, e):
    pass

  @abstractmethod
  def T(self, v, e):
    pass

  @abstractmethod
  def c(self, v, e):
    pass
