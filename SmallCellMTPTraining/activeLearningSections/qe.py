import os
import numpy as np
import subprocess
import math
import time
from SmallCellMTPTraining.templates import templates as templates
from SmallCellMTPTraining.templates import properties as properties
from SmallCellMTPTraining.io import writers as wr
from SmallCellMTPTraining.io import parsers as pa

def generateInitialDataset(inputFolder: str, outputFolder: str, config: dict):
    """
    Generates an initial dataset of strained atomic configurations using Quantum ESPRESSO (QE).
    
    Args:
        inputFolder (str): Directory to store QE input files and working directories.
        outputFolder (str): Directory to store QE output files.
        config (dict): Configuration dictionary containing parameters like strains, lattice constants, etc.
    
    Returns:
        list: Exit codes of all QE subprocesses.
    """
    # Generate a range of base strains (e.g., 0.9, 0.91, ..., 1.1 for 10% compression/expansion)
    baseStrains = np.arange(
        config["baseStrains"][0],  # Starting strain (e.g., 0.9 for 10% compression)
        config["baseStrains"][1],  # Ending strain (e.g., 1.1 for 10% expansion)
        config["baseStrainStep"],  # Step size (e.g., 0.01 for 1% increments)
    )

    # Parallel execution control
    maxCPUs = config["maxProcs"]  # Maximum allowed concurrent processes
    coresPerQE = 1                # CPUs per QE job (fixed to 1 here)
    cpusUsed = 0                  # Track active CPUs
    os.environ["OMP_NUM_THREADS"] = "1"  # Disable OpenMP parallelism (avoid oversubscription)
    subprocesses = []             # Store subprocess objects
    exitCodes = []                # Store exit codes
    completed = set()             # Track completed jobs

    # Loop over each strain value
    for strain in baseStrains:
        # Primitive job scheduler: Wait if CPU limit is reached
        if cpusUsed + coresPerQE > maxCPUs:
            available = False
            while not available:
                time.sleep(1)  # Check every second
                for i, p in enumerate(subprocesses):
                    if i in completed:
                        continue
                    exitCode = p.poll()  # Check if process finished
                    if exitCode is not None:
                        available = True
                        cpusUsed -= coresPerQE
                        completed.add(i)

        # Set up working directory and file paths
        workingFolder = os.path.join(inputFolder, "baseStrain" + str(round(strain, 3)))
        inputFile = os.path.join(workingFolder, "baseStrain" + str(round(strain, 3)) + ".in")
        outputFile = os.path.join(outputFolder, "baseStrain" + str(round(strain, 3)) + ".out")

        os.mkdir(workingFolder)  # Create a directory for this strain

        # Prepare QE input properties (atomic positions, lattice, etc.)
        qeProperties = {
            "atomPositions": np.array([
                [0, 0, 0],  # Atom 1 at origin
                [0.5. 0.5, 0],
                [0.5, 0, 0.5], 
                [0, 0.5, 0.5]              
            ])*config["baseLatticeParameter"] * strain #
            "atomTypes": [0, 0, 0, 0],  # Both atoms of type 0 (e.g., Silicon)
            "superCell": np.array([
                [strain * config["baseLatticeParameter"], 0, 0],  # Strained lattice vectors
                [0, strain * config["baseLatticeParameter"], 0],
                [0, 0, strain * config["baseLatticeParameter"]]
            ]),
            "kPoints": [10, 10, 10],  # k-point grid for Brillouin zone sampling
            "ecutwfc": 90,   # Wavefunction cutoff (Ry)
            "ecutrho": 450,  # Charge density cutoff (Ry)
            "qeOutDir": workingFolder,
            "elements": config["elements"],  # Chemical elements (e.g., ["Si"])
            "atomicWeights": config["atomicWeights"],
            "pseudopotentials": config["pseudopotentials"],
            "pseudopotentialDirectory": config["pseudopotentialDirectory"],
        }

        # Write QE input file and launch job
        wr.writeQEInput(inputFile, qeProperties)
        cpusUsed += coresPerQE
        subprocesses.append(
            subprocess.Popen(
                f"mpirun -np {coresPerQE} --bind-to none pw.x -in {inputFile} > {outputFile}",
                shell=True,
                cwd=workingFolder,
            )
        )

    # Wait for all jobs to finish and collect exit codes
    exitCodes = [p.wait() for p in subprocesses]
    return exitCodes


def calculateDiffConfigs(
    inputFolder: str,
    outputFolder: str,
    diffFile: str,
    attempt: str,
    stage: int,
    iteration: int,
    config: dict,
):
    """
    Runs QE simulations on perturbed atomic configurations (e.g., for MTP training).
    
    Args:
        inputFolder (str): Directory for QE input files.
        outputFolder (str): Directory for QE output files.
        diffFile (str): File containing perturbed atomic configurations.
        attempt (str): Identifier for the training attempt.
        stage (int): Current stage of the active learning loop.
        iteration (int): Current iteration within the stage.
        config (dict): Configuration parameters.
    
    Returns:
        tuple: (exit codes, CPU times spent for each job)
    """
    # Parse configurations from file
    newConfigs = pa.parsePartialMTPConfigsFile(diffFile)
    kPoints = config["kPoints"][stage]  # k-point grid may vary by stage
    maxCPUs = config["maxProcs"]
    coresPerQE = config["qeCPUsPerConfig"][stage]  # CPUs per job (can vary by stage)
    cpusUsed = 0
    os.environ["OMP_NUM_THREADS"] = "1"

    subprocesses = []
    exitCodes = []
    completed = set()
    cpuTimesSpent = []  # Track computational cost

    for j, newConfig in enumerate(newConfigs):
        # Job scheduler (same as in generateInitialDataset)
        if cpusUsed + coresPerQE > maxCPUs:
            available = False
            while not available:
                time.sleep(1)
                for i, p in enumerate(subprocesses):
                    if i in completed:
                        continue
                    exitCode = p.poll()
                    if exitCode is not None:
                        available = True
                        cpusUsed -= coresPerQE
                        completed.add(i)

        # Unique identifier for this job
        identifier = f"{attempt}_{stage}_{iteration}_{j}"
        workingFolder = os.path.join(inputFolder, identifier)
        os.mkdir(workingFolder)
        qeFile = os.path.join(workingFolder, identifier + ".in")
        outFile = os.path.join(outputFolder, identifier + ".out")

        # Prepare QE input properties
        qeProperties = {
            "atomPositions": newConfig["atomPositions"],
            "atomTypes": newConfig["atomTypes"],
            "superCell": newConfig["superCell"],
            "ecutrho": config["ecutrho"],
            "ecutwfc": config["ecutwfc"],
            "qeOutDir": workingFolder,
            "kPoints": kPoints,
            "elements": config["elements"],
            "atomicWeights": config["atomicWeights"],
            "pseudopotentials": config["pseudopotentials"],
            "pseudopotentialDirectory": config["pseudopotentialDirectory"],
        }

        # Launch QE job
        wr.writeQEInput(qeFile, qeProperties)
        cpusUsed += coresPerQE
        subprocesses.append(
            subprocess.Popen(
                f"mpirun -np {coresPerQE} --bind-to none pw.x -in {qeFile} > {outFile}",
                shell=True,
                cwd=workingFolder,
            )
        )

    exitCodes = [p.wait() for p in subprocesses]

    # Parse CPU times from QE output files
    for j, newConfig in enumerate(newConfigs):
        identifier = f"{attempt}_{stage}_{iteration}_{j}"
        outFile = os.path.join(outputFolder, identifier + ".out")
        qeOutput = pa.parseQEOutput(outFile)
        cpuTimesSpent.append(qeOutput["cpuTimeSpent"])

    return exitCodes, cpuTimesSpent


if __name__ == "__main__":
    """Entry point: Loads config.json and runs generateInitialDataset."""
    import json
    import shutil

    # Load configuration
    with open("./config.json", "r") as f:
        config = json.load(f)

    # Set up workspace
    if os.path.exists("./test"):
        shutil.rmtree("./test")
    os.mkdir("./test")

    # Run initial dataset generation
    generateInitialDataset(
        "/path/to/input_folder",
        "/path/to/output_folder",
        config,
    )
