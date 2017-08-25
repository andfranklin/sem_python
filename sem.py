"""
This script solves the seven-equation model (SEM) of 1-D two-phase flow
using the Continuous finite element method (CFEM). For help, please see
the external documentation in doc/.

Usage:
  sem.py -h | --help
  sem.py <INPUTFILE>

Options:
  -h --help  Display help information.

Arguments:
  INPUTFILE  Name of the input file
"""

from docopt import docopt

# base
from enums import ModelType
from Factory import Factory

# input
from InputFileParser import InputFileParser

# utilities
from error_utilities import error

## Runs the code with the given input file
# @param input_file  input file to run
# @param mods  input file modifications
def run(input_file, mods=list()):
  # parse the input file
  input_file_parser_original = InputFileParser()
  input_file_parser_original.parse(input_file)

  # check if the input file was a differential input file
  if input_file_parser_original.blockExists("BaseInputFile"):
    # get the name of the base input file and parse it
    base_input_file = input_file_parser_original.getBlockData("BaseInputFile")["base"]
    input_file_parser_base = InputFileParser()
    input_file_parser_base.parse(base_input_file)

    # apply modifications to base input file parser
    input_file_parser_base.applyDifferentialInputFileParser(input_file_parser_original)

    input_file_parser = input_file_parser_base
  else:
    input_file_parser = input_file_parser_original

  # apply modifications to input parameters, if any
  for mod in mods:
    input_file_parser.applyModification(mod)

  # create the factory
  factory = Factory()

  # model
  model_param_data = input_file_parser.getBlockData("Model")
  model = factory.createObject("Model", model_param_data)
  model_type = model.model_type

  # mesh(es)
  mesh_subblocks = input_file_parser.getSubblockNames("Mesh")
  if len(mesh_subblocks) == 0:
    # single mesh; no subblock is required
    mesh_param_data = input_file_parser.getBlockData("Mesh")
    mesh_class = mesh_param_data["type"]
    mesh = factory.createObject(mesh_class, mesh_param_data)
    meshes = [mesh]
  else:
    # multiple meshes; subblocks are required
    meshes = list()
    for mesh_subblock in mesh_subblocks:
      mesh_param_data = input_file_parser.getSubblockData("Mesh", mesh_subblock)
      mesh_class = mesh_param_data["type"]
      mesh_param_data["name"] = mesh_subblock
      mesh = factory.createObject(mesh_class, mesh_param_data)
      meshes.append(mesh)
  # check that no meshes have the same name
  mesh_names = list()
  for mesh in meshes:
    if mesh.name in mesh_names:
      error("Multiple meshes with the name '" + mesh.name + "'.")
    else:
      mesh_names.append(mesh.name)

  # equations of state
  eos_subblocks = input_file_parser.getSubblockNames("EoS")
  if model_type == ModelType.OnePhase:
    if len(eos_subblocks) != 1:
      error("Single-phase flow should have exactly 1 equation of state.")
  else:
    if len(eos_subblocks) != 2:
      error("Two-phase flow should have exactly 2 equations of state.")
  eos_list = list()
  phase_name_to_index = dict()
  for k, eos_subblock in enumerate(eos_subblocks):
    phase_name_to_index[eos_subblock] = k
    eos_param_data = input_file_parser.getSubblockData("EoS", eos_subblock)
    eos_class = eos_param_data["type"]
    eos_list.append(factory.createObject(eos_class, eos_param_data))

  # initial conditions / initial guess
  ic_param_data = input_file_parser.getBlockData("IC")
  if model_type == ModelType.OnePhase:
    ics = factory.createObject("InitialConditions1Phase", ic_param_data)
  else:
    ics = factory.createObject("InitialConditions2Phase", ic_param_data)

  # DoF handler
  dof_handler_params = {"meshes": meshes}
  if model_type == ModelType.OnePhase:
    dof_handler_class = "DoFHandler1Phase"
  elif model_type == ModelType.TwoPhaseNonInteracting:
    dof_handler_class = "DoFHandler2PhaseNonInteracting"
    dof_handler_params["initial_vf1"] = ics.vf1
  elif model_type == ModelType.TwoPhase:
    dof_handler_class = "DoFHandler2Phase"
  dof_handler = factory.createObject(dof_handler_class, dof_handler_params)

  # boundary conditions
  bcs = list()
  bc_subblocks = input_file_parser.getSubblockNames("BC")
  for bc_subblock in bc_subblocks:
    bc_param_data = input_file_parser.getSubblockData("BC", bc_subblock)
    bc_class = bc_param_data["type"]

    # if there is only 1 mesh, add the mesh parameter if not supplied already
    if len(meshes) == 1:
      if "mesh_name" not in bc_param_data:
        bc_param_data["mesh_name"] = meshes[0].name

    if "phase" in bc_param_data:
      bc_param_data["phase"] = phase_name_to_index[bc_param_data["phase"]]
    bc_param_data["dof_handler"] = dof_handler
    bc_param_data["eos_list"] = eos_list

    bc = factory.createObject(bc_class, bc_param_data)
    bcs.append(bc)

  # junctions
  junctions = list()
  if input_file_parser.blockExists("Junctions"):
    junction_subblocks = input_file_parser.getSubblockNames("Junctions")
    for junction_subblock in junction_subblocks:
      junction_param_data = input_file_parser.getSubblockData("Junctions", junction_subblock)
      junction_class = junction_param_data["type"]
      if "phase" in junction_param_data:
        junction_param_data["phase"] = phase_name_to_index[junction_param_data["phase"]]
      junction_param_data["dof_handler"] = dof_handler
      junction_param_data["eos_list"] = eos_list

      junction = factory.createObject(junction_class, junction_param_data)
      junctions.append(junction)

  # interface closures
  if model_type == ModelType.TwoPhase:
    interface_closures_params = input_file_parser.getBlockData("InterfaceClosures")
    interface_closures_params["factory"] = factory
    interface_closures_class = interface_closures_params["type"]
    interface_closures = factory.createObject(interface_closures_class, interface_closures_params)
  else:
    interface_closures = None

  # gravity
  physics_param_data = input_file_parser.getBlockData("Physics")
  physics_params = factory.createParametersObject("Physics", physics_param_data)
  gravity = physics_params.get("gravity")

  # nonlinear solver options
  nonlinear_solver_params = input_file_parser.getBlockData("NonlinearSolver")

  # stabilization
  if input_file_parser.blockExists("Stabilization"):
    stabilization_param_data = input_file_parser.getBlockData("Stabilization")
    stabilization_param_data["factory"] = factory
    stabilization_param_data["dof_handler"] = dof_handler
    stabilization_param_data["model_type"] = model_type
    stabilization_class = stabilization_param_data["type"]
  else:
    stabilization_param_data = {"factory": factory, "dof_handler": dof_handler, "model_type": model_type}
    stabilization_class = "NoStabilization"
  stabilization = factory.createObject(stabilization_class, stabilization_param_data)

  # create and run the executioner
  executioner_param_data = input_file_parser.getBlockData("Executioner")
  executioner_type = executioner_param_data["type"]
  executioner_param_data["model"] = model
  executioner_param_data["ics"] = ics
  executioner_param_data["bcs"] = bcs
  executioner_param_data["junctions"] = junctions
  executioner_param_data["eos_list"] = eos_list
  executioner_param_data["interface_closures"] = interface_closures
  executioner_param_data["gravity"] = gravity
  executioner_param_data["dof_handler"] = dof_handler
  executioner_param_data["meshes"] = meshes
  executioner_param_data["nonlinear_solver_params"] = nonlinear_solver_params
  executioner_param_data["stabilization"] = stabilization
  executioner_param_data["factory"] = factory
  executioner = factory.createObject(executioner_type, executioner_param_data)
  U = executioner.run()

  # create and run the postprocessor
  postprocessor_param_data = input_file_parser.getBlockData("Output")
  postprocessor_param_data["model"] = model
  postprocessor_param_data["eos_list"] = eos_list
  postprocessor_param_data["dof_handler"] = dof_handler
  postprocessor_param_data["meshes"] = meshes
  postprocessor = factory.createObject("Postprocessor", postprocessor_param_data)
  postprocessor.run(U)

## Gets the input file from the command line and runs it
def main():
  # parse command-line arguments using docopt
  command_line_arguments = docopt(__doc__)

  # get the name of the input file from the command line
  input_file = command_line_arguments["<INPUTFILE>"]

  # run with the input file
  run(input_file)

if __name__ == "__main__":
  main()
