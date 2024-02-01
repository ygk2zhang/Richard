import os
import numpy as np
import shutil

from SmallCellMTPTraining.activeLearningSections.initialDataset import (
    generateInitialDataset,
)
from SmallCellMTPTraining.io import writers as wr
from SmallCellMTPTraining.activeLearningSections.fileMovement import (
    compileTrainingConfigurations,
)
from SmallCellMTPTraining.activeLearningSections.mlip import trainMTP


def runActiveLearningScheme(rootFolder: str, config: dict):
    ### ===== Start by setting folders and file in the root folder =====
    rootFolder = os.path.abspath(rootFolder)  # Work in absolute paths where possible
    mtpFolder = os.path.join(rootFolder, "mtp")
    mdRunsFolder = os.path.join(rootFolder, "mdRuns")
    dftInputsFolder = os.path.join(rootFolder, "dftInputs")
    dftOutputsFolder = os.path.join(rootFolder, "dftOutputs")
    dftArchiveFolder = os.path.join(rootFolder, "dftArchive")
    logsFolder = os.path.join(rootFolder, "logs")

    for folder in [
        rootFolder,
        mtpFolder,
        mdRunsFolder,
        dftInputsFolder,
        dftOutputsFolder,
        dftArchiveFolder,
        logsFolder,
    ]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    # Logs and Stats
    logFile = os.path.join(logsFolder, "al.log")
    statsFile = os.path.join(logsFolder, "stats.log")

    # Setup the MTP file names
    mtpFile = os.path.join(mtpFolder, "pot.almtp")
    trainingFile = os.path.join(mtpFolder, "train.cfg")
    diffFile = os.path.join(mtpFolder, "diff.cfg")

    ### ===== Initalization of the base training set =====
    # Only generate a new initial dataset if there aren't existing dft files
    if len(os.listdir(dftArchiveFolder)) == 0:
        initalizationExitCodes = generateInitialDataset(
            dftInputsFolder, dftOutputsFolder, config
        )
        if bool(sum(initalizationExitCodes)):
            wr.printAndLog(logFile, "Initial DFT has failed. Exiting....")
            exit(1)
        compileTrainingConfigurations(
            trainingFile, dftOutputsFolder, dftArchiveFolder, "baseline"
        )

    ### ===== Prepare Active Learning Loop =====

    # Read the breadth of the parallel training runs
    boxDimensions = config["MDLatticeConfigs"]
    temperatures = config["MDTemperatures"]
    strains = np.arange(
        config["MDStrainRange"][0], config["MDStrainRange"][1], config["MDStrainStep"]
    )
    print(boxDimensions)
    # ### ===== Begin Active Learning Loop =====
    for boxDimension in boxDimensions:  # Iterate over all the configuration level
        hasPreselected = False
        # Iterate until we stop seeing preselcted configurations for the level
        while not hasPreselected:
            ### ===== Train the MTP ====
            # trainMTP(os.path.join(logsFolder,"train.qsub"),logsFolder,mtpFile,trainingFile)

            ### ===== Perform the MD active learning runs
            for temperature in temperatures:
                for strain in strains:
                    pass
