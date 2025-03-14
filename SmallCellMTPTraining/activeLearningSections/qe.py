import os
import numpy as np

# import regex as re  # Removed: regex no longer needed
import subprocess
import math
import time


from SmallCellMTPTraining.templates import templates as templates
from SmallCellMTPTraining.templates import properties as properties
from SmallCellMTPTraining.io import writers as wr
from SmallCellMTPTraining.io import parsers as pa


def generateInitialDataset(inputFolder: str, outputFolder: str, config: dict):
    # Extract the base data sets from the configurations
    baseStrains = np.arange(
        config["baseStrains"][0],
        config["baseStrains"][1],
        config["baseStrainStep"],
    )

    maxCPUs = config["maxProcs"]
    coresPerQE = 1
    cpusUsed = 0
    os.environ["OMP_NUM_THREADS"] = "1"
    subprocesses = []
    exitCodes = []
    completed = set()

    # Generate and submit the hydrostatic strain runs
    for strain in baseStrains:

        # This is essentially a primitive scheduler/semaphore
        if cpusUsed + coresPerQE > maxCPUs:
            available = False
            while not available:
                time.sleep(1)
                for i, p in enumerate(subprocesses):
                    if i in completed:
                        continue
                    exitCode = p.poll()
                    if not exitCode == None:
                        available = True
                        cpusUsed -= coresPerQE
                        completed.add(i)

        workingFolder = os.path.join(inputFolder, "baseStrain" + str(round(strain, 3)))
        inputFile = os.path.join(
            workingFolder, "baseStrain" + str(round(strain, 3)) + ".in"
        )
        outputFile = os.path.join(
            outputFolder, "baseStrain" + str(round(strain, 3)) + ".out"
        )

        os.mkdir(workingFolder)

        # Prepare QE input properties
        qeProperties = {
            "atomPositions": np.array(
                [
                    [0, 0, 0],
                    [
                        0.5 * config["baseLatticeParameter"] * strain,
                        0.5 * config["baseLatticeParameter"] * strain,
                        0.5 * config["baseLatticeParameter"] * strain,
                    ],
                ]
            ),
            "atomTypes": [0, 0],
            "superCell": np.array(
                [
                    [strain * config["baseLatticeParameter"], 0, 0],
                    [0, strain * config["baseLatticeParameter"], 0],
                    [0, 0, strain * config["baseLatticeParameter"]],
                ]
            ),
            "kPoints": [10, 10, 10],
            "ecutwfc": 90,
            "ecutrho": 450,
            "qeOutDir": workingFolder,
            "elements": config["elements"],
            "atomicWeights": config["atomicWeights"],
            "pseudopotentials": config["pseudopotentials"],
            "pseudopotentialDirectory": config["pseudopotentialDirectory"],
        }

        # Write the input and run
        wr.writeQEInput(inputFile, qeProperties)
        cpusUsed += coresPerQE
        subprocesses.append(
            subprocess.Popen(
                "mpirun -np "
                + str(coresPerQE)
                + " --bind-to none pw.x -in "
                + inputFile
                + " > "
                + outputFile,
                shell=True,
                cwd=workingFolder,
            ),
        )

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
    newConfigs = pa.parsePartialMTPConfigsFile(diffFile)
    kPoints = config["kPoints"][stage]
    maxCPUs = config["maxProcs"]
    coresPerQE = config["qeCPUsPerConfig"][stage]
    cpusUsed = 0
    os.environ["OMP_NUM_THREADS"] = "1"

    subprocesses = []
    exitCodes = []
    completed = set()

    for j, newConfig in enumerate(newConfigs):

        # This is essentially a primitive scheduler/semaphore
        if cpusUsed + coresPerQE > maxCPUs:
            available = False
            while not available:
                time.sleep(1)
                for i, p in enumerate(subprocesses):
                    if i in completed:
                        continue
                    exitCode = p.poll()
                    if not exitCode == None:
                        available = True
                        cpusUsed -= coresPerQE
                        completed.add(i)

        identifier = (
            str(attempt) + "_" + str(stage) + "_" + str(iteration) + "_" + str(j)
        )
        workingFolder = os.path.join(inputFolder, identifier)
        os.mkdir(workingFolder)
        qeFile = os.path.join(workingFolder, identifier + ".in")
        outFile = os.path.join(outputFolder, identifier + ".out")

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

        wr.writeQEInput(qeFile, qeProperties)
        cpusUsed += coresPerQE
        subprocesses.append(
            subprocess.Popen(
                "mpirun -np "
                + str(coresPerQE)
                + " --bind-to none pw.x -in "
                + qeFile
                + " > "
                + outFile,
                shell=True,
                cwd=workingFolder,
            ),
        )

    exitCodes = [p.wait() for p in subprocesses]

    cpuTimesSpent = []
    for j, newConfig in enumerate(newConfigs):
        identifier = (
            str(attempt) + "_" + str(stage) + "_" + str(iteration) + "_" + str(j)
        )
        outFile = os.path.join(outputFolder, identifier + ".out")
        qeOutput = pa.parseQEOutput(outFile)
        cpuTimesSpent.append(qeOutput["cpuTimeSpent"])

    return exitCodes, cpuTimesSpent


if __name__ == "__main__":
    import json
    import shutil

    with open("./config.json", "r") as f:
        out = json.load(f)
        # print(out)
        if os.path.exists("./test"):
            shutil.rmtree("./test")
        os.mkdir("./test")
        generateInitialDataset(
            "/global/home/hpc5146/Projects/SmallCellMTPTraining/SmallCellMTPTraining/activeLearning/test",
            "/global/home/hpc5146/Projects/SmallCellMTPTraining/SmallCellMTPTraining/activeLearning/testout",
            out,
        )
