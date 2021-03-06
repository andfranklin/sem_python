from filecmp import cmp
import sys
import unittest

from display_utilities import printDoFVector
from Factory import Factory
from OutputCaptor import OutputCaptor

def outputPrintDoFVectorToFile(path):
  # area function
  def A(x):
    return 2.0

  # create the DoF handler
  factory = Factory()
  n_cell = 5
  mesh = factory.createObject("UniformMesh", {"name": "mesh", "n_cell": n_cell})
  dof_handler = factory.createObject("DoFHandler1Phase", {"meshes": [mesh], "A": A})

  # solution vector
  U = range((n_cell + 1)*3)

  # capture the output
  captor = OutputCaptor()
  printDoFVector(U, dof_handler)
  out = captor.getCapturedOutput()

  # write the output file
  text_file = open(path + "print_dof_vector.txt", "w")
  text_file.write(out)
  text_file.close()

class DisplayUtilitiesTester(unittest.TestCase):
  def testPrintDoFVector(self):
    path = "tests/utilities/"
    outputPrintDoFVectorToFile(path)
    self.assertTrue(cmp(path + "print_dof_vector.txt", path + "gold/print_dof_vector.txt"))

if __name__ == "__main__":
  outputPrintDoFVectorToFile("")
