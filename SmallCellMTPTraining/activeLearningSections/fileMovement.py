import shutil
import os
import numpy as np

from SmallCellMTPTraining.io import parsers as pa
from SmallCellMTPTraining.io import writers as wr


def loadUntrainedMTP(mtpFile: str, untrainedMTPLevel: str):
    """
    Loads an untrained MTP potential file if it doesn't already exist.
    
    Args:
        mtpFile (str): Destination path where the MTP file should be saved.
        untrainedMTPLevel (str): Level of the untrained potential (e.g., 'low', 'medium', 'high').
    
    Description:
        - Checks if the target MTP file exists.
        - If not, copies a predefined untrained potential from the 'untrainedPots' directory.
    """
    if not os.path.exists(mtpFile):
        # Copy from predefined untrained potentials if target file doesn't exist
        shutil.copyfile(
            os.path.join(
                os.path.dirname(__file__),  # Get directory of current script
                "untrainedPots",             # Subdirectory containing untrained potentials
                untrainedMTPLevel + ".almtp"  # Filename format (e.g., 'low.almtp')
            ),
            mtpFile,  # Destination path
        )


def compileTrainingConfigurations(
    trainingFile: str,
    trainingConfigsFolder: str,
    archiveConfigsFolder: str,
    tag: str,
    config: dict,
):
    """
    Processes Quantum ESPRESSO (QE) outputs for MTP training and archives configurations.
    
    Args:
        trainingFile (str): Output file to store compiled MTP training data.
        trainingConfigsFolder (str): Directory containing QE output files.
        archiveConfigsFolder (str): Directory to archive processed configurations.
        tag (str): Identifier for the current training batch (used for archiving).
        config (dict): Configuration dictionary with reference energies and other parameters.
    
    Steps:
        1. Parses QE outputs and adjusts atom type indices.
        2. Normalizes energies using reference values.
        3. Archives processed configurations.
        4. Appends data to the training file.
    """
    # Parse all QE output files in the training directory
    qeOutputs = pa.parseAllQEInDirectory(trainingConfigsFolder)

    # Convert atom type indices from 1-based to 0-based (common in computational chemistry)
    for output in qeOutputs:
        for i, atomType in enumerate(output["atomTypes"]):
            output["atomTypes"][i] -= 1

    # Normalize energies using reference energies per atom type
    for output in qeOutputs:
        # Count atoms of each type (e.g., [5, 3] for 5 type-0 and 3 type-1 atoms)
        typeCounts = np.bincount(
            output["atomTypes"],
            minlength=len(config["baseEnergyReferences"])  # Ensure length matches reference energies
        )

        # Calculate total reference energy contribution
        totalReferenceEnergy = 0
        for i, count in enumerate(typeCounts):
            totalReferenceEnergy -= config["baseEnergyReferences"][i] * count

        # Adjust the QE energy by adding the reference energy
        output["energy"] += totalReferenceEnergy

    # Archive processed configurations and clear the working directory
    shutil.copytree(trainingConfigsFolder, os.path.join(archiveConfigsFolder, tag))
    shutil.rmtree(trainingConfigsFolder)
    os.mkdir(trainingConfigsFolder)  # Recreate empty directory for future runs

    # Write configurations to a temporary file, then append to main training file
    tempTrainingFile = trainingFile + ".temp"
    wr.writeMTPConfigs(tempTrainingFile, qeOutputs)  # Custom writer for MTP format

    # Append temporary file contents to the main training file
    with open(trainingFile, "a") as dest:      # Open in append mode
        with open(tempTrainingFile, "r") as src:
            dest.write(src.read())

    os.remove(tempTrainingFile)  # Clean up temporary file
