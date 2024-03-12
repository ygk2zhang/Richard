import os
import shutil
import regex as re
import subprocess
import numpy as np
from SmallCellMTPTraining.io import parsers as pa

trainingJobTemplate = """#!/bin/bash
#SBATCH --account=def-hpcg1725
#SBATCH --partition=reserved
#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=train
#SBATCH --output=./$num.out
#SBATCH --time=00-16:00 # time (DD-HH:MM)
#SBATCH --mem-per-cpu=4G
#SBATCH --wait

module load       StdEnv/2020  gcc/9.3.0  cuda/11.2.2
module load openmpi/4.0.3

mpirun -np 4 --oversubscribe /global/home/hpc5146/mlip-2/bin/mlp train ../pot.mtp ../train.cfg --max-iter=10000 --bfgs-conv-tol=0.000001 --trained-pot-name=$out
"""


def generateEnsembleOfMTPs(
    mtpLevel: str,
    trainingConfigurationsFile: str,
    mtpEnsembleLocation: str,
    ensembleSize: int,
    mlpBinary="/global/home/hpc5146/mlip-3/bin/mlp",
):
    # Clear the existing folder if applicable and generate the ensemble folder
    if os.path.exists(mtpEnsembleLocation):
        shutil.rmtree(mtpEnsembleLocation)
    os.mkdir(mtpEnsembleLocation)

    baseMTPFile = os.path.join(
        os.path.dirname(__file__),
        "untrainedPots",
        mtpLevel + ".almtp",
    )

    shutil.copy(baseMTPFile, mtpEnsembleLocation + "/" + "pot.mtp")
    shutil.copy(trainingConfigurationsFile, mtpEnsembleLocation + "/" + "train.cfg")

    # Generate the final mtp location
    finalMTPFolder = mtpEnsembleLocation + "/mtps"
    os.mkdir(finalMTPFolder)

    # Generate the job files folder
    jobsFolder = mtpEnsembleLocation + "/jobs"
    os.mkdir(jobsFolder)
    os.chdir(jobsFolder)

    subprocesses = []

    for i in range(ensembleSize):  # For each member of the ensemble
        # Get the names
        outputName = "../mtps/" + str(i) + ".mtp"
        jobName = str(i) + ".qsub"

        # Replace the output name in the job folder and write the job folder
        trainingJobText = re.sub("\$out", outputName, trainingJobTemplate)
        trainingJobText = re.sub("\$num", str(i), trainingJobText)
        with open(jobName, "w") as jobFile:
            jobFile.write(trainingJobText)

        subprocesses.append(subprocess.Popen(["sbatch", jobName]))

    exitCodes = [p.wait() for p in subprocesses]  # Wait for all the diffDFT to finish
    subprocesses = []

    averageEnergyErrorsPerAtom = []
    averageForceErrorsPerAtom = []

    for i in range(ensembleSize):
        jobOutput = str(i) + ".out"
        with open(jobOutput, "r") as txtfile:
            lines = txtfile.readlines()
            for i, line in enumerate(lines):
                if line == "Energy per atom:\n":
                    averageEnergyErrorsPerAtom.append(float(lines[i + 3][31:-1]))
                if line == "Forces:\n":
                    averageForceErrorsPerAtom.append(float(lines[i + 3][31:-1]))

    bestPerformer = np.argmin(averageEnergyErrorsPerAtom)

    if os.path.exists("../stats.txt"):
        os.remove("../stats.txt")
    with open("../stats.txt", "a") as file:
        file.write(
            "Average training energy error per atom "
            + str(np.mean(averageEnergyErrorsPerAtom))
            + "\n"
        )
        file.write(
            "Average training force error per atom "
            + str(np.mean(averageForceErrorsPerAtom))
            + "\n"
        )
        file.write(
            "STD training energy error per atom "
            + str(np.std(averageEnergyErrorsPerAtom))
            + "\n"
        )
        file.write(
            "STD training force error per atom "
            + str(np.std(averageForceErrorsPerAtom))
            + "\n"
        )
        file.write(
            "Lowest Energy Error " + str(np.min(averageEnergyErrorsPerAtom)) + "\n"
        )
        file.write("Lowest Energy Error MTP " + str(bestPerformer) + "\n")

    shutil.copy("../mtps/" + str(bestPerformer) + ".mtp", "../best.mtp")


def predictUsingEnsemble(
    mtpEnsembleLocation: str,
    configurationsToEvaluate,
    parallelize=False,
    mlpBinary="/global/home/hpc5146/mlip-3/bin/mlp",
):
    mtpFolder = mtpEnsembleLocation + "/mtps"
    outputsFolder = mtpEnsembleLocation + "/outputs"
    if os.path.exists(outputsFolder):
        shutil.rmtree(outputsFolder)
    os.mkdir(outputsFolder)

    results = []

    for mtpFile in os.listdir(mtpFolder):
        os.system(
            mlpBinary
            + " calculate_efs "
            + os.path.join(mtpFolder, mtpFile)
            + " "
            + configurationsToEvaluate
            + " --output_filename="
            + os.path.join(outputsFolder, mtpFile)
        )
        results.append(
            pa.parseMTPConfigsFile(
                os.path.join(outputsFolder, mtpFile + ".0"), convertFromAngRy=False
            )
        )

    return results


def predictUsingBest(
    mtpEnsembleLocation: str,
    configurationsToEvaluate,
    parallelize=False,
    mlpBinary="/global/home/hpc5146/mlip-3/bin/mlp",
):
    outputsFolder = mtpEnsembleLocation + "/outputs"

    if os.path.exists(outputsFolder):
        shutil.rmtree(outputsFolder)
    os.mkdir(outputsFolder)

    os.system(
        mlpBinary
        + " calculate_efs "
        + os.path.join(mtpEnsembleLocation, "best.mtp")
        + " "
        + configurationsToEvaluate
        + " --output_filename="
        + os.path.join(outputsFolder, "best.mtp")
    )
    return [
        pa.parseMTPConfigsFile(
            os.path.join(outputsFolder, "best.mtp" + ".0"), convertFromAngRy=False
        )
    ]


if __name__ == "__main__":
    generateEnsembleOfMTPs("10.almtp", "train.cfg", "test", 7)
