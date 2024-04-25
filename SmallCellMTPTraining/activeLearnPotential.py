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

    # Used for tags are restarting runs
    attempt = 1
    startingStage = 0
    startingIter = 0

    ### ===== Determine if we are continuing a existing run by reading the DFT Archive =====
    existingArchiveFolders = os.listdir(dftArchiveFolder)
    # Only generate a new initial dataset if there aren't existing dft files
    if len(existingArchiveFolders) == 0:
        ### ===== Initialization of the base training set =====
        wr.printAndLog(logFile, "Generating Initial Dataset.")
        initializationExitCodes = generateInitialDataset(
            dftInputsFolder, dftOutputsFolder, config
        )
        if bool(sum(initializationExitCodes)):
            wr.printAndLog(logFile, "Initial DFT has failed. Exiting....")
            exit(1)
        compileTrainingConfigurations(
            trainingFile, dftOutputsFolder, dftArchiveFolder, "baseline", config
        )
        wr.printAndLog(logFile, "Completed Initial Dataset Generation.")

    # Otherwise, we are continuing a run, and we should parse where we were
    elif (
        len(existingArchiveFolders) == 1
    ):  # This only happens if we fail right after the baseline
        attempt += 1
    else:
        # In order to extract where to resume the runs from, we parse the dft archive
        attempt, startingStage, startingIter = 0, 0, 0
        maxScore = 0
        for archive in existingArchiveFolders:
            if archive == "baseline":
                continue
            vec = tuple(map(int, archive.split("_")))
            score = (
                vec[0] * 1e6 + vec[1] * 1e3 + vec[2]
            )  # We calculate a score based on the importance of the measure

            # Store the archive with the attempt, state, and iteration that has the highest score
            if score > maxScore:
                attempt, startingStage, startingIter = tuple(
                    map(int, archive.split("_"))
                )  # Extract
                maxScore = score

        # Increment these by one to continue form where we left off
        attempt += 1
        startingIter += 1

    wr.printAndLog(logFile, "=======================================================")
    wr.printAndLog(
        logFile, "Starting Active Learning Process !!! Attempt: " + str(attempt)
    )
    wr.printAndLog(logFile, "=======================================================")

    ### ===== Prepare Active Learning Loop =====
    # Read the breadth of the parallel training runs
    cellDimensionsList = config["mdLatticeConfigs"]
    temperatures = config["mdTemperatures"]
    strains = np.arange(
        config["mdStrainRange"][0], config["mdStrainRange"][1], config["mdStrainStep"]
    )

    # ### ===== Begin Active Learning Loop =====
    for stage in range(
        startingStage, len(cellDimensionsList), 1
    ):  # Iterate over all the configuration level
        cellDimensions = cellDimensionsList[stage]
        # Iterate until we stop seeing preselcted configurations for the level
        for iteration in range(startingIter, config["maxItersPerConfig"][stage], 1):

            if not startingIter == 0:  # Clear the startingIter after setting it once
                startingIter = 0

            wr.printAndLog(
                logFile,
                "Entering "
                + str(cellDimensions)
                + "; iteration "
                + str(iteration)
                + ".",
            )

            ### ===== Train the MTP ====
            wr.printAndLog(logFile, "Starting Training Stage.")
            avgEnergyError, avgForceError, trainingTime = trainMTP(
                os.path.join(logsFolder, "train.qsub"),
                logsFolder,
                potFile,
                trainingFile,
            )
            wr.printAndLog(
                logFile,
                "Training Stage Complete; Wall Time: " + str(trainingTime) + "s",
            )
            wr.printAndLog(logFile, "Average Energy Per Atom Error: " + avgEnergyError)
            wr.printAndLog(logFile, "Average Force Per Atom Error: " + avgForceError)

            ### ===== Perform the MD active learning runs
            wr.printAndLog(logFile, "Starting MD Runs.")
            (
                mdExitCodes,
                preselectedIterationLogs,
                temperatureAverageGrades,
                hasPreselected,
                mdCPUTimesSpent,
            ) = performParallelMDRuns(
                temperatures,
                strains,
                stage,
                mdRunsFolder,
                potFile,
                masterPreselectedFile,
                config,
            )
            wr.printAndLog(
                logFile,
                "MD Calculations Complete.\tTotal CPU seconds spent this iteration: "
                + str(round(np.sum(mdCPUTimesSpent), 2))
                + "s.\tLimiting CPU time: "
                + str(round(np.max(mdCPUTimesSpent), 2))
                + "s.",
            )
            if not hasPreselected:
                preselectedLogs.append(preselectedIterationLogs)
                wr.printAndLog(
                    logFile,
                    "No Preselected Found. Going to Next Lattice Configuration.",
                )
                break

            wr.printAndLog(
                logFile,
                "Completed MD Runs. Average Extrapolation Grade By Temperature: "
                + str(temperatureAverageGrades),
            )

            ### ===== Select the needed configurations =====
            wr.printAndLog(logFile, "Selecting New Configurations.")
            diffCount, meanGrade, maxGrade, selectTime = selectDiffConfigs(
                os.path.join(logsFolder, "selectAdd.qsub"),
                logsFolder,
                potFile,
                trainingFile,
                masterPreselectedFile,
                diffFile,
            )
            wr.printAndLog(
                logFile, "Selection Completed; Wall time: " + str(selectTime) + "s"
            )
            wr.printAndLog(
                logFile,
                str(diffCount)
                + " New Configurations Found.     Average Grade: "
                + str(round(meanGrade, 2))
                + "    Max Grade: "
                + str(round(maxGrade, 2)),
            )
            os.remove(masterPreselectedFile)

            ### ===== Calculate the Needed Configurations Using Quantum Espresso =====
            qeExitCodes, dftCPUTimesSpent = calculateDiffConfigs(
                dftInputsFolder,
                dftOutputsFolder,
                diffFile,
                attempt,
                stage,
                iteration,
                config,
            )
            wr.printAndLog(
                logFile,
                "DFT Calculations Complete.\tTotal CPU seconds spent this iteration: "
                + str(round(np.sum(dftCPUTimesSpent), 2))
                + "s.\tLimiting CPU time: "
                + str(round(np.max(dftCPUTimesSpent), 2))
                + "s.",
            )
            ### ===== Clean up the DFT outputs by putting them into the archive =====
            compileTrainingConfigurations(
                trainingFile,
                dftOutputsFolder,
                dftArchiveFolder,
                str(attempt) + "_" + str(stage) + "_" + str(iteration),
                config,
            )

            pass
