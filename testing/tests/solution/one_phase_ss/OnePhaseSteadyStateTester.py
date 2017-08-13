import unittest
import filecmp

import sem
from InputFileModification import InputFileModification

class OnePhaseSteadyStateTester(unittest.TestCase):
  ## Tests the solution
  def testSolution(self):
    test_dir = "tests/solution/one_phase_ss/"
    sem.run(test_dir + "one_phase_ss.in")
    self.assertTrue(filecmp.cmp(test_dir + "solution.csv", test_dir + "gold/solution.csv"))

if __name__ == "__main__":
  mods = list()
  mods.append(InputFileModification("NonlinearSolver", "verbose", True))
  mods.append(InputFileModification("Output", "save_solution", False))
  sem.run("one_phase_ss.in", mods)
