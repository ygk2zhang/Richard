import os
import numpy as np
import regex as re
import subprocess
import math
import time


from SmallCellMTPTraining.templates import templates as templates
from SmallCellMTPTraining.templates import properties as properties
from SmallCellMTPTraining.io import writers as wr
from SmallCellMTPTraining.io import parsers as pa


def calculateDiffConfigs(
    inputFolder: str,
    outputFolder: str,
    diffFile: str,
    attempt: str,
    iteration: int,
    config: dict,
):
    newConfigs = pa.parsePartialMTPConfigsFile(diffFile)
    kPoints = config["kPoints"]
    subprocesses = []

    for j, newConfig in enumerate(newConfigs):
        identifier = str(attempt) + "_" + str(iteration) + "_" + str(j)
        workingFolder = os.path.join(inputFolder, identifier)
        os.mkdir(workingFolder)
        qeFile = os.path.join(workingFolder, identifier + ".in")
        runFile = os.path.join(workingFolder, identifier + ".run")
        outFile = os.path.join(outputFolder, identifier + ".out")
        jobFile = os.path.join(workingFolder, identifier + ".qsub")

        qeProperties = {
            "atomPositions": newConfig["atomPositions"],
            "atomTypes": newConfig["atomTypes"],
            "superCell": newConfig["superCell"],
            "ecutrho": config["ecutrho"],
            "ecutwfc": config["ecutwfc"],
            "qeOutDir": workingFolder,
            "kPoints": kPoints,
        }

        jobProperties = {
            "jobName": identifier,
            "ncpus": config["qeCPUs"],
            "memPerCpu": config["qeMem"],
            "maxDuration": config["qeTime"],
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
        identifier = str(attempt) + "_" + str(iteration) + "_" + str(j)
        outFile = os.path.join(outputFolder, identifier + ".out")
        try:
            qeOutput = pa.parseQEOutput(outFile)
        except:
            continue
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
        # generateInitialDataset(
        #     "/global/home/hpc5146/Projects/SmallCellMTPTraining/SmallCellMTPTraining/activeLearning/test",
        #     "/global/home/hpc5146/Projects/SmallCellMTPTraining/SmallCellMTPTraining/activeLearning/testout",
        #     out,
        # )
