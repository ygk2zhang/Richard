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


def generateInitialDataset(inputFolder: str, outputFolder: str, params: dict):
    # Extract the base data sets from the configurations
    baseStrains = np.arange(
        params["baseStrains"][0],
        params["baseStrains"][1],
        params["baseStrainStep"],
    )

    jobProperties = {
        "ncpus": 1,
        "maxDuration": "0-1:20",
        "memPerCpu": "8G",
    }

    subprocesses = []

    # Generate and submit the 1Atom Strain runs
    for strain in baseStrains:
        workingFolder = os.path.join(inputFolder, "baseStrain" + str(round(strain, 3)))
        inputFile = os.path.join(
            workingFolder, "baseStrain" + str(round(strain, 3)) + ".in"
        )
        jobFile = os.path.join(
            workingFolder, "baseStrain" + str(round(strain, 3)) + ".qsub"
        )
        runFile = os.path.join(
            workingFolder, "baseStrain" + str(round(strain, 3)) + ".run"
        )
        outputFile = os.path.join(
            outputFolder, "baseStrain" + str(round(strain, 3)) + ".out"
        )

        os.mkdir(workingFolder)

        # Prepare QE input properties
        qeProperties = {
            "atomPositions": np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
            "atomTypes": [0],
            "superCell": np.array(
                [
                    [strain * params["baseLatticeParameter"], 0, 0],
                    [0, strain * params["baseLatticeParameter"], 0],
                    [0, 0, strain * params["baseLatticeParameter"]],
                ]
            ),
            "kPoints": [12, 12, 12],
            "ecutwfc": 90,
            "ecutrho": 450,
            "qeOutDir": workingFolder,
            "elements": params["elements"],
            "atomicWeights": params["atomicWeights"],
            "pseudopotentials": params["pseudopotentials"],
        }
        # Prepare the job file
        jobProperties["jobName"] = "baseStrain" + str(round(strain, 3))
        jobProperties["inFile"] = inputFile
        jobProperties["outFile"] = outputFile
        jobProperties["runFile"] = runFile

        # Write the input and the run file
        wr.writeQEInput(inputFile, qeProperties)  # Use writeQEInput
        wr.writeQEJob(jobFile, jobProperties)

        subprocesses.append(subprocess.Popen(["sbatch", jobFile]))

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
    cellDimensions = config["mdLatticeConfigs"][stage]
    kPoints = config["kPoints"][stage]
    subprocesses = []

    for j, newConfig in enumerate(newConfigs):
        identifier = (
            str(attempt) + "_" + str(stage) + "_" + str(iteration) + "_" + str(j)
        )
        workingFolder = os.path.join(inputFolder, identifier)
        os.mkdir(workingFolder)
        qeFile = os.path.join(workingFolder, identifier + ".in")
        runFile = os.path.join(workingFolder, identifier + ".run")
        outFile = os.path.join(outputFolder, identifier + ".out")
        jobFile = os.path.join(workingFolder, identifier + ".qsub")
        # Added element information
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
        }

        jobProperties = {
            "jobName": identifier,
            "ncpus": config["qeCPUsPerConfig"][stage],
            "memPerCpu": config["qeMemPerConfig"][stage],
            "maxDuration": config["qeTimePerConfig"][stage],
            "inFile": qeFile,
            "outFile": outFile,
            "runFile": runFile,
        }

        wr.writeQEInput(qeFile, qeProperties)
        wr.writeQEJob(jobFile, jobProperties)
        subprocesses.append(subprocess.Popen(["sbatch", jobFile]))
        time.sleep(0.1)

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
