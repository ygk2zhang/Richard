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


def runActiveLearningScheme(
    rootFolder: str, config: dict, mtpLevel="08", initial_train=None, initial_pot=None
):

    ### ===== Start by setting folders and file in the root folder =====
    rootFolder = os.path.abspath(rootFolder)  # Work in absolute paths where possible
    mtpFolder = os.path.join(rootFolder, "mtp")
    mdRunsFolder = os.path.join(rootFolder, "mdRuns")
    dftInputsFolder = os.path.join(rootFolder, "dftInputs")
    dftOutputsFolder = os.path.join(rootFolder, "dftOutputs")
    dftArchiveFolder = os.path.join(rootFolder, "dftArchive")
    logsFolder = os.path.join(rootFolder, "logs")
    oldPotFolder = os.path.join(rootFolder, "oldPot")

    for folder in [
        rootFolder,
        mtpFolder,
        mdRunsFolder,
        dftInputsFolder,
        dftOutputsFolder,
        dftArchiveFolder,
        logsFolder,
        oldPotFolder,
    ]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    # Logs and Stats
    logFile = os.path.join(logsFolder, "al.log")
    statsFile = os.path.join(logsFolder, "stats.log")
    preselectedLogs = []

    # Setup the MTP file names
    potFile = os.path.join(mtpFolder, "pot.almtp")
    trainingFile = os.path.join(mtpFolder, "train.cfg")
    masterPreselectedFile = os.path.join(mtpFolder, "preselected.cfg")
    diffFile = os.path.join(mtpFolder, "diff.cfg")

    # Used for tags are restarting runs
    attempt = 1
    startingStage = 0
    startingIter = 0

    ### ===== Determine if we are continuing a existing run by reading the DFT Archive =====
    existingArchiveFolders = os.listdir(dftArchiveFolder)
    is_resuming = len(existingArchiveFolders) > 0  # Check if we are resuming

    if not initial_train and initial_pot and not is_resuming:
        raise ValueError(
            "Unless resuming a run, you must provide an initial training set if an initial potential is provided!"
        )

    if initial_train and not is_resuming:
        if not os.path.exists(initial_train):  # Check that the inital train exists
            raise FileNotFoundError(
                f"Initial potential file not found: {initial_train}"
            )
        wr.printAndLog(logFile, f"Using initial potential/dataset: {initial_train}")
        shutil.copyfile(
            initial_train, trainingFile
        )  # Copy the specified initial training set
        if initial_pot:
            shutil.copyfile(
                initial_pot, potFile
            )  # Copy the specified initial training pot
        else:
            loadUntrainedMTP(potFile, mtpLevel)

    if not initial_train and not is_resuming:
        if len(config["elements"]) > 1:
            raise ValueError(
                "For multiple elements, an initial training set (`initial_train`) MUST be provided."
            )
        # Single element, new run, no initial training set
        loadUntrainedMTP(potFile, mtpLevel)
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
    if is_resuming:
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
            if initial_pot:
                initial_pot = None
            else:
                wr.printAndLog(logFile, "Starting Training Stage.")
                avgEnergyError, avgForceError, trainingTime = trainMTP(
                    os.path.join(logsFolder, "train.qsub"),
                    logsFolder,
                    potFile,
                    trainingFile,
                    config,
                )
                wr.printAndLog(
                    logFile,
                    "Training Stage Complete; CPU Time: "
                    + str(round(trainingTime, 2))
                    + "s",
                )
                wr.printAndLog(
                    logFile, "Average Energy Per Atom Error: " + avgEnergyError
                )
                wr.printAndLog(
                    logFile, "Average Force Per Atom Error: " + avgForceError
                )
                # Store an archive of the last trained potential
                shutil.copyfile(
                    potFile,
                    os.path.join(
                        oldPotFolder,
                        str(attempt) + "_" + str(stage) + "_" + str(iteration),
                    ),
                )

            ### ===== Perform the MD active learning runs
            wr.printAndLog(logFile, "Starting MD Runs.")
            (
                mdExitCodes,
                preselectedIterationLogs,
                hasPreselected,
                mdCPUTimesSpent,
            ) = performParallelMDRuns(
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
            if sum(mdExitCodes) > 0 and not hasPreselected:
                raise RuntimeError("MD run failed! Check MTP for sufficient species.")
            if not hasPreselected:
                preselectedLogs.append(preselectedIterationLogs)
                wr.printAndLog(
                    logFile,
                    "No Preselected Found. Going to Next Lattice Configuration.",
                )
                break

            ### ===== Select the needed configurations =====
            wr.printAndLog(logFile, "Selecting New Configurations.")
            diffCount, meanGrade, maxGrade, selectTime = selectDiffConfigs(
                os.path.join(logsFolder, "selectAdd.qsub"),
                logsFolder,
                potFile,
                trainingFile,
                masterPreselectedFile,
                diffFile,
                config,
            )
            wr.printAndLog(
                logFile,
                "Selection Completed; CPU time: " + str(round(selectTime, 2)) + "s",
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
            wr.printAndLog(logFile, "Starting new DFT calculations!")
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
                "DFT Calculations Complete. Used "
                + str(config["qeCPUsPerConfig"][stage])
                + " cores per job. \tTotal CPU seconds spent this iteration: "
                + str(
                    round(
                        np.sum(dftCPUTimesSpent) * config["qeCPUsPerConfig"][stage], 2
                    )
                )
                + "s.\tLimiting CPU time: "
                + str(
                    round(
                        np.max(dftCPUTimesSpent) * config["qeCPUsPerConfig"][stage], 2
                    )
                )
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
