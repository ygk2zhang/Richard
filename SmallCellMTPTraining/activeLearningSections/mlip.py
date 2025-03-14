import os
import subprocess
import regex as re
import numpy as np

from SmallCellMTPTraining.io import writers as wr
from SmallCellMTPTraining.io import parsers as pa


def trainMTP(
    jobFile: str, logsFolder: str, potFile: str, trainingFIle: str, config: dict
):
    runFile = os.path.join(logsFolder, "train.out")
    timeFile = os.path.join(logsFolder, "train.time")

    maxCPUs = config["maxProcs"]

    subprocess.Popen(
        "/usr/bin/time -o "
        + timeFile
        + ' -f "%e" mpirun -np '
        + str(maxCPUs)
        + " --bind-to none --oversubscribe "
        + config["mlpBinary"]
        + " train "
        + potFile
        + " "
        + trainingFIle
        + " --iteration_limit=1000 --tolerance=0.000001 --init_random=false --al_mode="
        + config["mode"]
        + " > "
        + runFile,
        shell=True,
    ).wait()

    with open(runFile, "r") as txtfile:
        lines = txtfile.readlines()
        for i, line in enumerate(lines):
            if line == "Energy per atom:\n":
                avgEnergyError = lines[i + 3][31:-1]
            if line == "Forces:\n":
                avgForceError = lines[i + 3][31:-1]

    timeSpent = pa.parseTimeFile(timeFile) * maxCPUs

    return avgEnergyError, avgForceError, timeSpent


def selectDiffConfigs(
    jobFile: str,
    logsFolder: str,
    potFile: str,
    trainingFile: str,
    preselectedFile: str,
    diffFile: str,
    config: str,
):
    timeFile = os.path.join(logsFolder, "selectAdd.time")
    maxCPUs = min(config["maxProcs"], 12)

    subprocess.Popen(
        [
            "/usr/bin/time",
            "-o",
            timeFile,
            "-f",
            "%e",
            "mpirun",
            "-np",
            str(maxCPUs),
            "--oversubscribe",
            "--bind-to",
            "none",
            config["mlpBinary"],
            "select_add",
            potFile,
            trainingFile,
            preselectedFile,
            diffFile,
        ],
        stdout=subprocess.PIPE,
    ).wait()

    timeSpent = pa.parseTimeFile(timeFile) * maxCPUs

    with open(diffFile, "r") as f:
        content = f.read()
        preselectedGrades = list(
            map(float, re.findall(r"(?<=MV_grade\t)\d+.?\d*", content))
        )
        return (
            len(preselectedGrades),
            np.mean(preselectedGrades),
            np.max(preselectedGrades),
            timeSpent,
        )
