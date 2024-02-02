import shutil
import os
from SmallCellMTPTraining.io import parsers as pa
from SmallCellMTPTraining.io import writers as wr


def loadUntrainedMTP(mtpFile: str, untrainedMTPLevel: str):
    if not os.path.exists(mtpFile):
        shutil.copyfile(
            os.path.join(
                os.path.dirname(__file__),
                "untrainedPots",
                untrainedMTPLevel + ".almtp",
            ),
            mtpFile,
        )


def compileTrainingConfigurations(
    trainingFile: str,
    trainingConfigsFolder: str,
    archiveConfigsFolder: str,
    tag: str,
    config: dict,
):
    qeOutputs = pa.parseAllQEInDirectory(
        trainingConfigsFolder
    )  # Extract the Quantum Espresso Configs

    # Change the type index from 1-indexing to 0 indexing and rescale the energies
    for output in qeOutputs:
        for i, atomType in enumerate(output["atomTypes"]):
            output["atomTypes"][i] -= 1
        output["energy"] -= config["baseEnergyReference"] * len(output["atomTypes"])

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
