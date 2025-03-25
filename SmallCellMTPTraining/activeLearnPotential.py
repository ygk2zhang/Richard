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
    rootFolder: str, 
    config: dict, 
    mtpLevel="08", 
    initial_train=None, 
    initial_pot=None
):
    """
    Main function to run the active learning scheme for MTP potential development.
    
    Args:
        rootFolder (str): Root directory for all outputs and working files
        config (dict): Configuration dictionary containing all parameters
        mtpLevel (str): Level of untrained MTP to use if no initial potential provided
        initial_train (str): Path to initial training set file (optional)
        initial_pot (str): Path to initial potential file (optional)
    """

    ### ===== Directory Setup =====
    # Convert to absolute paths and create all required directories
    rootFolder = os.path.abspath(rootFolder)
    
    # Define all necessary subdirectories
    mtpFolder = os.path.join(rootFolder, "mtp")          # For MTP-related files
    mdRunsFolder = os.path.join(rootFolder, "mdRuns")    # Molecular dynamics runs
    dftInputsFolder = os.path.join(rootFolder, "dftInputs")  # DFT input files
    dftOutputsFolder = os.path.join(rootFolder, "dftOutputs") # DFT outputs
    dftArchiveFolder = os.path.join(rootFolder, "dftArchive") # Archived DFT data
    logsFolder = os.path.join(rootFolder, "logs")        # Log files
    oldPotFolder = os.path.join(rootFolder, "oldPot")    # Previous potentials
    
    # Create directories if they don't exist
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

    ### ===== File Path Setup =====
    # Logging files
    logFile = os.path.join(logsFolder, "al.log")         # Main log file
    statsFile = os.path.join(logsFolder, "stats.log")    # Statistics file
    preselectedLogs = []                                 # Store MD selection logs
    
    # MTP-related files
    potFile = os.path.join(mtpFolder, "pot.almtp")               # Current MTP potential
    trainingFile = os.path.join(mtpFolder, "train.cfg")          # Training configurations
    masterPreselectedFile = os.path.join(mtpFolder, "preselected.cfg") # MD-selected configs
    diffFile = os.path.join(mtpFolder, "diff.cfg")               # Configurations needing DFT

    ### ===== Run Initialization/Resumption =====
    attempt = 1                  # Attempt counter
    startingStage = 0            # Starting stage index
    startingIter = 0             # Starting iteration index
    
    # Check if we're resuming an existing run by looking for archived DFT data
    existingArchiveFolders = os.listdir(dftArchiveFolder)
    is_resuming = len(existingArchiveFolders) > 0
    
    # Validate input combinations
    if not initial_train and initial_pot and not is_resuming:
        raise ValueError(
            "Unless resuming a run, you must provide an initial training set if an initial potential is provided!"
        )

    # Handle initial training data and potential
    if initial_train and not is_resuming:
        if not os.path.exists(initial_train):
            raise FileNotFoundError(f"Initial potential file not found: {initial_train}")
        
        wr.printAndLog(logFile, f"Using initial potential/dataset: {initial_train}")
        shutil.copyfile(initial_train, trainingFile)  # Copy initial training set
        
        if initial_pot:
            shutil.copyfile(initial_pot, potFile)     # Copy initial potential
        else:
            loadUntrainedMTP(potFile, mtpLevel)        # Load default untrained potential

    # For new runs with no initial data
    if not initial_train and not is_resuming:
        if len(config["elements"]) > 1:
            raise ValueError(
                "For multiple elements, an initial training set (`initial_train`) MUST be provided."
            )
        
        # Single element case - generate initial dataset
        loadUntrainedMTP(potFile, mtpLevel)
        wr.printAndLog(logFile, "Generating Initial Dataset.")
        
        # Generate initial strained configurations with DFT
        initializationExitCodes = generateInitialDataset(
            dftInputsFolder, dftOutputsFolder, config
        )
        
        if bool(sum(initializationExitCodes)):
            wr.printAndLog(logFile, "Initial DFT has failed. Exiting....")
            exit(1)
            
        # Process and archive the initial DFT data
        compileTrainingConfigurations(
            trainingFile, dftOutputsFolder, dftArchiveFolder, "baseline", config
        )
        wr.printAndLog(logFile, "Completed Initial Dataset Generation.")

    # Handle resuming existing run
    if is_resuming:
        # Parse archive folders to determine where to resume
        attempt, startingStage, startingIter = 0, 0, 0
        maxScore = 0
        
        for archive in existingArchiveFolders:
            if archive == "baseline":
                continue
                
            # Extract attempt, stage, iteration from folder names (format "attempt_stage_iteration")
            vec = tuple(map(int, archive.split("_")))
            
            # Calculate a score to find the most advanced run
            score = vec[0] * 1e6 + vec[1] * 1e3 + vec[2]  
            
            if score > maxScore:
                attempt, startingStage, startingIter = vec
                maxScore = score

        # Increment to continue from next step
        attempt += 1
        startingIter += 1

    ### ===== Active Learning Loop =====
    wr.printAndLog(logFile, "=======================================================")
    wr.printAndLog(logFile, "Starting Active Learning Process !!! Attempt: " + str(attempt))
    wr.printAndLog(logFile, "=======================================================")

    # Get the list of MD cell dimensions for each stage
    cellDimensionsList = config["mdLatticeConfigs"]
    
    # Main loop over all stages
    for stage in range(startingStage, len(cellDimensionsList), 1):
        cellDimensions = cellDimensionsList[stage]
        
        # Loop over iterations within each stage
        for iteration in range(startingIter, config["maxItersPerConfig"][stage], 1):
            
            # Reset starting iteration after first use
            if not startingIter == 0:
                startingIter = 0

            wr.printAndLog(
                logFile,
                "Entering " + str(cellDimensions) + "; iteration " + str(iteration) + ".",
            )

            ### === MTP Training ===
            if initial_pot:
                initial_pot = None  # Only use initial pot once
            else:
                wr.printAndLog(logFile, "Starting Training Stage.")
                
                # Train the MTP potential
                avgEnergyError, avgForceError, trainingTime = trainMTP(
                    os.path.join(logsFolder, "train.qsub"),
                    logsFolder,
                    potFile,
                    trainingFile,
                    config,
                )
                
                wr.printAndLog(
                    logFile,
                    "Training Stage Complete; CPU Time: " + str(round(trainingTime, 2)) + "s",
                )
                wr.printAndLog(logFile, "Average Energy Per Atom Error: " + avgEnergyError)
                wr.printAndLog(logFile, "Average Force Per Atom Error: " + avgForceError)
                
                # Archive the trained potential
                shutil.copyfile(
                    potFile,
                    os.path.join(
                        oldPotFolder,
                        str(attempt) + "_" + str(stage) + "_" + str(iteration),
                    ),
                )

            ### === Molecular Dynamics Runs ===
            wr.printAndLog(logFile, "Starting MD Runs.")
            
            # Perform MD to find candidate configurations
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
            
            # Check for MD failures
            if sum(mdExitCodes) > 0 and not hasPreselected:
                raise RuntimeError("MD run failed! Check MTP for sufficient species.")
                
            # Break if no configurations were selected
            if not hasPreselected:
                preselectedLogs.append(preselectedIterationLogs)
                wr.printAndLog(
                    logFile,
                    "No Preselected Found. Going to Next Lattice Configuration.",
                )
                break

            ### === Configuration Selection ===
            wr.printAndLog(logFile, "Selecting New Configurations.")
            
            # Select configurations needing DFT calculations
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
            os.remove(masterPreselectedFile)  # Clean up

            ### === DFT Calculations ===
            wr.printAndLog(logFile, "Starting new DFT calculations!")
            
            # Perform DFT on selected configurations
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
                + str(round(np.sum(dftCPUTimesSpent) * config["qeCPUsPerConfig"][stage], 2))
                + "s.\tLimiting CPU time: "
                + str(round(np.max(dftCPUTimesSpent) * config["qeCPUsPerConfig"][stage], 2))
                + "s.",
            )
            
            ### === Data Processing ===
            # Archive and incorporate new DFT data
            compileTrainingConfigurations(
                trainingFile,
                dftOutputsFolder,
                dftArchiveFolder,
                str(attempt) + "_" + str(stage) + "_" + str(iteration),
                config,
            )
