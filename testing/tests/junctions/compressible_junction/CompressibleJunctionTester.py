import unittest

import sem
from ParameterModification import BlockParameterModification
from CSVTester import CSVTester
from JunctionTester import JunctionTester

class CompressibleJunctionTester(unittest.TestCase):
  def runDerivativeTest(self, test_option, use_momentum_flux_balance):
    tester = JunctionTester("CompressibleJunction")
    matched = tester.checkJacobian(test_option,
      junction_params={"use_momentum_flux_balance": use_momentum_flux_balance})
    n_i, n_j = matched.shape
    for i in xrange(n_i):
      for j in xrange(n_j):
        self.assertTrue(matched[i,j])

  def testJacobianWeakTechnique1(self):
    self.runDerivativeTest("weak", False)

  def testJacobianStrongTechnique1(self):
    self.runDerivativeTest("strong", False)

  def testJacobianWeakTechnique2(self):
    self.runDerivativeTest("weak", True)

  def testJacobianStrongTechnique2(self):
    self.runDerivativeTest("strong", True)

if __name__ == "__main__":
  tester = JunctionTester("CompressibleJunction", verbose=True)
  print "\nTECHNIQUE 1: Stagnation pressure"
  _ = tester.checkJacobian("weak", junction_params={"use_momentum_flux_balance": False})
  _ = tester.checkJacobian("strong", junction_params={"use_momentum_flux_balance": False})
  print "\nTECHNIQUE 2: Momentum flux balance"
  _ = tester.checkJacobian("weak", junction_params={"use_momentum_flux_balance": True})
  _ = tester.checkJacobian("strong", junction_params={"use_momentum_flux_balance": True})
