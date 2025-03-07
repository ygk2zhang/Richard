import os
import shutil
import subprocess
import random
import numpy as np
import regex as re
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

    os.environ["OMP_NUM_THREADS"] = "1"

    maxCPUs = len(os.sched_getaffinity(0)) - 1
    cellDimensions = config["mdLatticeConfigs"][i]
    hasPreselected = False
    subprocesses = []
    temperatures = []
    strains = []
    identifiers = []
    exitCodes = []

    seen_identifiers = set()

    max_attempts = 10
    attempts = 0

    maxTempThreshold = max(
        6, int(maxCPUs / 4)
    )  # Run a fourth of them at max temp, at least 6

    # This guarentees no repeated temperature/ strains
    for j in range(maxCPUs):
        while attempts < max_attempts:
            attempts += 1
            temperature = random.uniform(
                config["mdTemperatureRange"][0], config["mdTemperatureRange"][1]
            )
            if j < maxTempThreshold:
                temperature = config["mdTemperatureRange"][1]
            strain = random.uniform(
                config["mdStrainRange"][0], config["mdStrainRange"][1]
            )
            rounded_temperature = round(temperature, 0)
            rounded_strain = round(strain, 3)

            identifier = (
                "".join(str(x) for x in cellDimensions)
                + "_T"
                + str(rounded_temperature)
                + "_S"
                + str(rounded_strain)
            )

            if identifier not in seen_identifiers:
                seen_identifiers.add(identifier)
                identifiers.append(identifier)
                temperatures.append(temperature)
                strains.append(strain)
                break
        else:
            raise ValueError(
                f"Failed to generate unique parameters after {max_attempts} attempts."
            )
        attempts = 0

    for j in range(maxCPUs):
        temperature = temperatures[j]
        strain = strains[j]
        identifier = identifiers[j]

        workingFolder = os.path.join(mdFolder, identifier)
        os.mkdir(workingFolder)
        mdFile = os.path.join(workingFolder, identifier + ".in")
        outFile = os.path.join(workingFolder, identifier + ".out")
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

        wr.writeMDInput(mdFile, mdProperties)
        subprocesses.append(
            subprocess.Popen(
                [
                    "/usr/bin/time",
                    "-o",
                    timeFile,
                    "-f",
                    "%e",
                    "mpirun",
                    "-np",
                    "1",
                    "--bind-to",
                    "none",
                    config["lmpMPIFile"],
                    "-in",
                    mdFile,
                    "-log",
                    outFile,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workingFolder,
            )
        )

    exitCodes = [p.wait() for p in subprocesses]
    for exitCode in exitCodes:
        if exitCode != 0 and exitCode != 9:
            raise RuntimeError("MD runs have failed!")
    preselectedIterationLogs = {}
    cpuTimesSpent = []

    for j in range(maxCPUs):
        temperature = temperatures[j]
        strain = strains[j]
        identifier = identifiers[j]
        workingFolder = os.path.join(mdFolder, identifier)
        preselectedFile = os.path.join(workingFolder, "preselected.cfg.0")

        # Record the time spent
        timeFile = os.path.join(workingFolder, identifier + ".time")
        cpuTimesSpent.append(pa.parseTimeFile(timeFile))

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

    return (
        exitCodes,
        preselectedIterationLogs,
        hasPreselected,
        cpuTimesSpent,
    )
