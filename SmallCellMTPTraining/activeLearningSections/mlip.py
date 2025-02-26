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
    jobProperties = {
        "jobName": "train",
        "ncpus": config["trainCPUs"],
        "runFile": runFile,
        "timeFile": timeFile,
        "maxDuration": config["trainTime"],
        "memPerCpu": "4G",
        "potFile": potFile,
        "trainFile": trainingFIle,
        "initRandom": "false",
        "mode": config["mode"],
    }

    wr.writeTrainJob(jobFile, jobProperties)
    subprocess.Popen(["sbatch", jobFile]).wait()

    with open(runFile, "r") as txtfile:
        lines = txtfile.readlines()
        for i, line in enumerate(lines):
            if line == "Energy per atom:\n":
                avgEnergyError = lines[i + 4][31:-1]
            if line == "Forces:\n":
                avgForceError = lines[i + 4][31:-1]

    timeSpent = pa.parseTimeFile(timeFile)

    return avgEnergyError, avgForceError, timeSpent


def selectDiffConfigs(
    jobFile: str,
    logsFolder: str,
    potFile: str,
    trainingFile: str,
    preselectedFile: str,
    diffFile: str,
):
    timeFile = os.path.join(logsFolder, "selectAdd.time")
    jobProperties = {
        "jobName": "selectAdd",
        "ncpus": 6,
        "runFile": os.path.join(logsFolder, "selectAdd.out"),
        "timeFile": timeFile,
        "maxDuration": "0-8:00",
        "memPerCpu": "6G",
        "potFile": potFile,
        "trainFile": trainingFile,
        "preselectedFile": preselectedFile,
        "diffFile": diffFile,
    }

    wr.writeSelectJob(jobFile, jobProperties)
    subprocess.Popen(["sbatch", jobFile]).wait()

    timeSpent = pa.parseTimeFile(timeFile)

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
