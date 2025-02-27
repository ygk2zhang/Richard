import os
import shutil
import subprocess
import random
import numpy as np
import time

import SmallCellMTPTraining.io.writers as wr
import SmallCellMTPTraining.io.parsers as pa


def performParallelMDRuns(
    i: int,
    mdFolder: str,
    potFile: str,
    masterPreselectedFile: str,
    config: dict,
):
    # Remove and remake folder to get new preselected, Slightly inefficient
    shutil.rmtree(mdFolder)
    os.mkdir(mdFolder)

    cellDimensions = config["mdLatticeConfigs"][i]
    hasPreselected = False
    subprocesses = []
    temperatures = []
    strains = []
    identifiers = []

    for i in range(config["parallelMDRuns"]):
        temperature = random.uniform(
            config["mdTemperatureRange"][0], config["mdTemperatureRange"][1]
        )
        strain = random.uniform(config["mdStrainRange"][0], config["mdStrainRange"][1])
        identifiers.append(
            "".join(str(x) for x in cellDimensions)
            + "_T"
            + str(round(temperature, 3))
            + "_S"
            + str(round(strain, 3))
        )
        temperatures.append(temperature)
        strains.append(strain)

    for i in range(config["parallelMDRuns"]):
        temperature = temperatures[i]
        strain = strains[i]
        identifier = identifiers[i]

        workingFolder = os.path.join(mdFolder, identifier)
        os.mkdir(workingFolder)
        mdFile = os.path.join(workingFolder, identifier + ".in")
        runFile = os.path.join(workingFolder, identifier + ".run")
        outFile = os.path.join(workingFolder, identifier + ".out")
        jobFile = os.path.join(workingFolder, identifier + ".qsub")
        timeFile = os.path.join(workingFolder, identifier + ".time")

        latticeParameter = config["baseLatticeParameter"] * strain

        mdProperties = {
            "latticeParameter": latticeParameter,
            "temperature": temperature,
            "potFile": potFile,
            "boxDimensions": config["mdLatticeConfigs"][i],
            "elements": config["elements"],
            "atomicWeights": config["atomicWeights"],
        }

        jobProperties = {
            "jobName": identifier,
            "ncpus": config["mdCPUsPerConfig"][i],
            "memPerCpu": config["mdMemPerConfig"][i],
            "maxDuration": config["mdTimePerConfig"][i],
            "inFile": mdFile,
            "outFile": outFile,
            "runFile": runFile,
            "timeFile": timeFile,
        }

        wr.writeMDInput(mdFile, mdProperties)
        wr.writeMDJob(jobFile, jobProperties)
        subprocesses.append(subprocess.Popen(["sbatch", jobFile]))
        time.sleep(0.1)

    exitCodes = [p.wait() for p in subprocesses]

    preselectedIterationLogs = {}
    cpuTimesSpent = []

    for i in range(config["parallelMDRuns"]):
        temperature = temperatures[i]
        strain = strains[i]
        identifier = identifiers[i]
        workingFolder = os.path.join(mdFolder, identifier)
        preselectedFile = os.path.join(workingFolder, "preselected.cfg.0")

        # Record the time spent
        timeFile = os.path.join(workingFolder, identifier + ".time")
        cpuTimesSpent.append(pa.parseTimeFile(timeFile))

        preselectedGrades = []

        if os.path.exists(preselectedFile):
            hasPreselected = True
            preselected_configs = pa.parsePartialMTPConfigsFile(preselectedFile)
            preselectedGrades = [
                cfg["MV_grade"]
                for cfg in preselected_configs
                if cfg["MV_grade"] is not None
            ]

            with open(masterPreselectedFile, "a") as dest:
                wr.writeMTPConfigs(dest, preselected_configs)

        preselectedIterationLogs[identifier] = preselectedGrades

    return (
        exitCodes,
        preselectedIterationLogs,
        hasPreselected,
        cpuTimesSpent,
    )
