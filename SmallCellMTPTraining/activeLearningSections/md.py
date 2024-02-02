import os
import shutil
import subprocess
import regex as re
import numpy as np

import SmallCellMTPTraining.io.writers as wr


def performParallelMDRuns(
    temperatures: list,
    strains: list,
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

    for temperature in temperatures:
        for strain in strains:
            identifier = (
                "".join(str(x) for x in cellDimensions)
                + "T"
                + str(int(temperature))
                + "S"
                + str(round(strain, 2))
            )
            workingFolder = os.path.join(mdFolder, identifier)
            os.mkdir(workingFolder)
            mdFile = os.path.join(workingFolder, identifier + ".in")
            runFile = os.path.join(workingFolder, identifier + ".run")
            outFile = os.path.join(workingFolder, identifier + ".out")
            jobFile = os.path.join(workingFolder, identifier + ".qsub")

            latticeParameter = config["baseLatticeParameter"] * strain

            mdProperties = {
                "latticeParameter": latticeParameter,
                "temperature": temperature,
                "potFile": potFile,
                "boxDimensions": config["mdLatticeConfigs"][i],
            }

            jobProperties = {
                "jobName": identifier,
                "ncpus": config["mdCPUsPerConfig"][i],
                "memPerCpu": config["mdMemPerConfig"][i],
                "maxDuration": config["mdTimePerConfig"][i],
                "inFile": mdFile,
                "outFile": outFile,
                "runFile": runFile,
            }

            wr.writeMDInput(mdFile, mdProperties)
            wr.writeMDJob(jobFile, jobProperties)
            subprocesses.append(subprocess.Popen(["sbatch", jobFile]))

    exitCodes = [p.wait() for p in subprocesses]

    preselectedIterationLogs = {}
    temperatureGrades = {temperature: [] for temperature in temperatures}

    for temperature in temperatures:
        for strain in strains:
            identifier = (
                "".join(str(x) for x in cellDimensions)
                + "T"
                + str(int(temperature))
                + "S"
                + str(round(strain, 2))
            )
            workingFolder = os.path.join(mdFolder, identifier)
            preselectedFile = os.path.join(workingFolder, "preselected.cfg.0")

            preselectedGrades = []

            if os.path.exists(preselectedFile):
                hasPreselected = True
                with open(preselectedFile, "r") as src:
                    content = src.read()
                    preselectedGrades = list(
                        map(float, re.findall(r"(?<=MV_grade\t)\d+.?\d*", content))
                    )
                    with open(masterPreselectedFile, "a") as dest:
                        dest.write(content)

            preselectedIterationLogs[identifier] = preselectedGrades
            if len(preselectedGrades) != 0:
                temperatureGrades[temperature].append(np.mean(preselectedGrades))

    temperatureAverageGrades = {
        temperature: "No Preselected" for temperature in temperatures
    }

    for temperature, temperatureGrade in temperatureGrades.items():
        if len(temperatureGrade) > 0:
            temperatureAverageGrades[temperature] = round(np.mean(temperatureGrade), 2)

    return exitCodes, preselectedIterationLogs, temperatureAverageGrades, hasPreselected
