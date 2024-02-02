import os
import numpy as np
import shutil

from SmallCellMTPTraining.activeLearningSections.qe import (
    generateInitialDataset,
    calculateDiffConfigs,
)
from SmallCellMTPTraining.io import writers as wr
from SmallCellMTPTraining.activeLearningSections.fileMovement import (
    compileTrainingConfigurations,
    loadUntrainedMTP,
)
from SmallCellMTPTraining.activeLearningSections.mlip import trainMTP, selectDiffConfigs
from SmallCellMTPTraining.activeLearningSections.md import performParallelMDRuns


def runActiveLearningScheme(rootFolder: str, config: dict, mtpLevel="08"):
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
    preselectedLogs = []

    # Setup the MTP file names
    potFile = os.path.join(mtpFolder, "pot.almtp")
    loadUntrainedMTP(potFile, mtpLevel)
    trainingFile = os.path.join(mtpFolder, "train.cfg")
    masterPreselectedFile = os.path.join(mtpFolder, "preselected.cfg")
    diffFile = os.path.join(mtpFolder, "diff.cfg")

    wr.printAndLog(logFile, "=======================================================")
    wr.printAndLog(logFile, "Starting New Potential Active Learning Process !!!")
    wr.printAndLog(logFile, "=======================================================")

    ### ===== Initalization of the base training set =====
    # Only generate a new initial dataset if there aren't existing dft files

    if len(os.listdir(dftArchiveFolder)) == 0:
        wr.printAndLog(logFile, "Generating Inital Dataset.")
        initalizationExitCodes = generateInitialDataset(
            dftInputsFolder, dftOutputsFolder, config
        )
        if bool(sum(initalizationExitCodes)):
            wr.printAndLog(logFile, "Initial DFT has failed. Exiting....")
            exit(1)
        compileTrainingConfigurations(
            trainingFile, dftOutputsFolder, dftArchiveFolder, "baseline", config
        )
        wr.printAndLog(logFile, "Completed Inital Dataset Generation.")

    ### ===== Prepare Active Learning Loop =====

    # Read the breadth of the parallel training runs
    cellDimensionsList = config["mdLatticeConfigs"]
    temperatures = config["mdTemperatures"]
    strains = np.arange(
        config["mdStrainRange"][0], config["mdStrainRange"][1], config["mdStrainStep"]
    )

    # ### ===== Begin Active Learning Loop =====
    for i, cellDimensions in enumerate(
        cellDimensionsList
    ):  # Iterate over all the configuration level
        # Iterate until we stop seeing preselcted configurations for the level
        for iter in range(config["maxItersPerConfig"][i]):
            wr.printAndLog(
                logFile,
                "Entering " + str(cellDimensions) + "; iteration " + str(i) + ".",
            )

            ### ===== Train the MTP ====
            wr.printAndLog(logFile, "Starting Training Iteration.")
            avgEnergyError, avgForceError = trainMTP(
                os.path.join(logsFolder, "train.qsub"),
                logsFolder,
                potFile,
                trainingFile,
            )
            wr.printAndLog(logFile, "Training Iteration Completed.")
            wr.printAndLog(logFile, "Average Energy Per Atom Error: " + avgEnergyError)
            wr.printAndLog(logFile, "Average Force Per Atom Error: " + avgForceError)

            ### ===== Perform the MD active learning runs
            wr.printAndLog(logFile, "Starting MD Runs.")
            (
                mdExitCodes,
                preselectedIterationLogs,
                temperatureAverageGrades,
                hasPreselected,
            ) = performParallelMDRuns(
                temperatures,
                strains,
                i,
                mdRunsFolder,
                potFile,
                masterPreselectedFile,
                config,
            )
            if not hasPreselected:
                wr.printAndLog(
                    logFile,
                    "No Preselected Found. Going to Next Lattice Configuration.",
                )
                break
            preselectedLogs.append(preselectedIterationLogs)
            wr.printAndLog(
                logFile,
                "Completed MD Runs. Average Extrapolation Grade By Temp: "
                + str(temperatureAverageGrades),
            )

            ### ===== Select the needed configurations =====
            wr.printAndLog(logFile, "Selecting New Configurations.")
            diffCount, meanGrade, maxGrade = selectDiffConfigs(
                os.path.join(logsFolder, "selectAdd.qsub"),
                logsFolder,
                potFile,
                trainingFile,
                masterPreselectedFile,
                diffFile,
            )
            wr.printAndLog(
                logFile,
                str(diffCount)
                + " New Configurations Found.     Average Grade: "
                + str(meanGrade)
                + "    Max Grade: "
                + str(maxGrade),
            )
            os.remove(masterPreselectedFile)

            ### ===== Calculate the Needed Configurations Using Quantum Espresso ====
            calculateDiffConfigs(
                dftInputsFolder, dftOutputsFolder, diffFile, i, iter, config
            )
            wr.printAndLog(logFile, "DFT Calculations Complete")

            exit()
            pass
