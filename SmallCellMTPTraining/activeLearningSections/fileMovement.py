import shutil
import os
from SmallCellMTPTraining.io import parsers as pa
from SmallCellMTPTraining.io import writers as wr


def compileTrainingConfigurations(
    trainingFile: str, trainingConfigsFolder: str, archiveConfigsFolder: str, tag: str
):
    qeOutputs = pa.parseAllQEInDirectory(
        trainingConfigsFolder
    )  # Extract the Quantum Espresso Configs

    # Move the outputs to the archive and clear the folder
    shutil.copytree(trainingConfigsFolder, os.path.join(archiveConfigsFolder, tag))
    shutil.rmtree(trainingConfigsFolder)
    os.mkdir(trainingConfigsFolder)

    # Write the qeOutputs as a MTP as temporary training file
    tempTrainingFile = trainingFile + ".temp"
    wr.writeMTPConfigs(tempTrainingFile, qeOutputs)
    with open(trainingFile, "a") as dest:
        with open(tempTrainingFile, "r") as src:
            dest.write(src.read())
    os.remove(tempTrainingFile)
