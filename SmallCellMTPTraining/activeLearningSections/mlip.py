import os
import subprocess

from SmallCellMTPTraining.io import writers as wr


def trainMTP(jobFile: str, logsFolder: str, potFile: str, trainingFIle: str):
    jobProperties = {
        "jobName": "train",
        "ncpus": 12,
        "runFile": os.path.join(logsFolder, "train.out"),
        "maxDuration": "0-2:00",
        "memPerCpu": "2G",
        "potFile": potFile,
        "trainFile": trainingFIle,
        "initRandom": "false",
    }

    wr.writeTrainJob(jobFile, jobProperties)
    return subprocess.Popen(["sbatch", jobFile]).wait()


def selectDiffConfigs(
    jobFile: str,
    logsFolder: str,
    potFile: str,
    trainingFile: str,
    preselectedFile: str,
    diffFile: str,
):
    jobProperties = {
        "jobName": "selectADD",
        "ncpus": 2,
        "runFile": os.path.join(logsFolder, "train.out"),
        "maxDuration": "0-1:00",
        "memPerCpu": "6G",
        "potFile": potFile,
        "trainFile": trainingFile,
        "preselectedFile": preselectedFile,
        "diffFile": diffFile,
    }

    wr.writeTrainJob(jobFile, jobProperties)
    return subprocess.Popen(["sbatch", jobFile]).wait()
