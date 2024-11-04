import os
import shutil
import subprocess
import regex as re
import numpy as np
import time
import random

import SmallCellMTPTraining.io.writers as wr
import SmallCellMTPTraining.io.parsers as pa

startingPositionsCoords = """1 2 7.16001 12.2677 12.6308 0 0 0
2 1 4.84976 11.8173 13.3638 0 0 0
3 1 6.97683 14.3782 12.9683 0 0 0
4 1 5.22053 3.71588 13.8087 0 0 0
5 1 3.12975 0.14767 14.1531 0 0 0
6 1 7.47467 13.5578 8.21887 0 0 0
7 2 7.39075 12.0121 6.40066 0 0 0
8 2 5.45219 12.6051 9.07187 0 0 0
9 1 4.27075 11.5403 7.0226 0 0 0
10 1 4.47697 14.1702 7.25393 0 0 0
11 2 2.39112 13.6603 8.51984 0 0 0
12 2 2.59969 12.5492 4.72444 0 0 0
13 1 6.28406 2.87822 7.63074 0 0 0
14 2 8.0087 0.627825 6.74645 0 0 0
15 2 7.0921 4.20523 5.56922 0 0 0
16 1 2.77285 2.36157 7.95672 0 0 0
17 2 2.58568 1.54219 5.71147 0 0 0
18 2 4.92133 1.64759 9.35781 0 0 0
19 2 4.28767 4.27597 6.52642 0 0 0
20 1 7.39455 12.9085 1.72682 0 0 0
21 1 8.10357 13.7821 4.70174 0 0 0
22 2 7.58914 11.0192 2.93977 0 0 0
23 2 5.60724 14.2186 3.74112 0 0 0
24 1 4.3713 12.3637 2.75735 0 0 0
25 2 5.41204 13.6924 0.269486 0 0 0
26 2 1.8127 13.0585 0.763559 0 0 0
27 1 6.69214 4.07429 2.29242 0 0 0
28 2 7.94866 1.79742 2.87775 0 0 0
29 2 6.99008 2.38949 14.3071 0 0 0
30 1 5.95669 2.0716 6.07292 0 0 0
31 1 2.76336 14.6343 3.06927 0 0 0
32 1 6.14399 1.06489 1.79064 0 0 0
33 1 2.19032 3.58705 4.76499 0 0 0
34 2 3.25016 2.62202 0.598395 0 0 0
"""


def modifyTypeData(atom_string, from_types, to_type, num_transitions):
    """Randomly converts a specified number of atom types in a string."""

    lines = atom_string.strip().split("\n")
    atom_data = []
    for line in lines:
        if line.strip():
            atom_data.append(line.split())

    from_type_indices = [
        i for i, row in enumerate(atom_data) if int(row[1]) in from_types
    ]

    # Check if num_transitions exceeds the available atoms of the specified types
    if num_transitions > len(from_type_indices):
        raise ValueError(
            f"Requested {num_transitions} transitions, but only {len(from_type_indices)} atoms of the specified types are available."
        )

    selected_indices = random.sample(from_type_indices, num_transitions)

    for i in selected_indices:
        atom_data[i][1] = str(to_type)

    modified_lines = [" ".join(row) for row in atom_data]
    return "\n".join(modified_lines)


def performParallelMDRuns(
    mdFolder: str,
    potFile: str,
    masterPreselectedFile: str,
    config: dict,
):
    # Remove and remake folder to get new preselected, Slightly inefficient
    shutil.rmtree(mdFolder)
    os.mkdir(mdFolder)

    hasPreselected = False

    subprocesses = []

    for i in range(config["mdInstanceCount"]):
        temp = np.random.uniform(*config["temperatureRange"])
        press = np.random.uniform(*config["pressureRange"])
        replacementCount = np.random.randint(*config["replacementRange"])

        identifier = str(i)
        workingFolder = os.path.join(mdFolder, identifier)
        os.mkdir(workingFolder)
        mdFile = os.path.join(workingFolder, identifier + ".in")
        dataFile = os.path.join(workingFolder, "config.dat")
        runFile = os.path.join(workingFolder, identifier + ".run")
        outFile = os.path.join(workingFolder, identifier + ".out")
        jobFile = os.path.join(workingFolder, identifier + ".qsub")
        timeFile = os.path.join(workingFolder, identifier + ".time")

        dataTemplate = """LAMMPS data file via write_data, version 23 Jun 2022, timestep = 0

34 atoms
3 atom types

0 16.4934  xlo xhi
0 14.6317  ylo yhi
0 14.5809 zlo zhi

Masses

1 39.0983
2 35.453
3 15.999

Atoms # atomic

{data}"""
        dataContent = dataTemplate.format(
            data=modifyTypeData(startingPositionsCoords, [1, 2], 3, replacementCount)
        )

        mdProperties = {"potFile": potFile, "temperature": temp, "pressure": press}
        jobProperties = {
            "jobName": identifier,
            "ncpus": config["mdCPUs"],
            "memPerCpu": config["mdMem"],
            "maxDuration": config["mdTime"],
            "inFile": mdFile,
            "outFile": outFile,
            "runFile": runFile,
            "timeFile": timeFile,
        }

        with open(dataFile, "w") as f:
            f.write(dataContent)
        wr.writeMDInput(mdFile, mdProperties)
        wr.writeMDJob(jobFile, jobProperties)
        subprocesses.append(subprocess.Popen(["sbatch", jobFile]))
        time.sleep(0.1)

    exitCodes = [p.wait() for p in subprocesses]

    preselectedIterationLogs = {}
    cpuTimesSpent = []

    for i in range(config["mdInstanceCount"]):
        identifier = str(i)
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
