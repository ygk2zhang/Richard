import os
import shutil
import subprocess
import random
import numpy as np
import regex as re
import time

import SmallCellMTPTraining.io.writers as wr
import SmallCellMTPTraining.io.parsers as pa


def performParallelMDRuns(
    i: int,
    mdFolder: str,
    potFile: str,
    masterPreselectedFile: str,
    config: dict,
):
    """
    Performs parallel molecular dynamics (MD) runs to identify configurations needing DFT calculations.
    
    Args:
        i (int): Index of the current MD lattice configuration
        mdFolder (str): Directory to store MD run outputs
        potFile (str): Path to current MTP potential file
        masterPreselectedFile (str): File to collect all preselected configurations
        config (dict): Configuration dictionary containing MD parameters
        
    Returns:
        tuple: (exit codes, preselection logs, hasPreselected flag, CPU times)
    """
    
    # Clean and recreate MD folder to ensure fresh start
    shutil.rmtree(mdFolder)
    os.mkdir(mdFolder)

    # Set environment for single-threaded execution
    os.environ["OMP_NUM_THREADS"] = "1"
    maxCPUs = config["maxProcs"]  # Maximum number of parallel MD runs

    # Get current cell dimensions from config
    cellDimensions = config["mdLatticeConfigs"][i]
    hasPreselected = False  # Flag if any run produces preselected configs
    
    # Lists to track run parameters and processes
    subprocesses = []
    temperatures = []
    pressures = []
    identifiers = []
    exitCodes = []
    seen_identifiers = set()  # Ensure unique run parameters

    max_attempts = 10  # Maximum attempts to generate unique parameters
    attempts = 0

    # Calculate how many runs should use maximum temperature
    maxTempThreshold = max(6, int(maxCPUs / 4))  # At least 6, up to 1/4 of maxCPUs

    # Generate unique temperature/pressure combinations for each run
    for j in range(maxCPUs):
        while attempts < max_attempts:
            attempts += 1
            # Random temperature within range (use max temp for first few runs)
            temperature = random.uniform(
                config["mdTemperatureRange"][0], config["mdTemperatureRange"][1]
            )
            if j < maxTempThreshold:
                temperature = config["mdTemperatureRange"][1]  # Use max temp
                
            # Random pressure within range
            pressure = random.uniform(
                config["mdPressureRange"][0], config["mdPressureRange"][1]
            )
            
            # Create unique identifier for this run
            rounded_temperature = round(temperature)
            rounded_pressure = round(pressure)
            identifier = (
                "".join(str(x) for x in cellDimensions)
                + "_T" + str(rounded_temperature)
                + "_S" + str(rounded_pressure)
            )

            # Ensure we haven't used these parameters before
            if identifier not in seen_identifiers:
                seen_identifiers.add(identifier)
                identifiers.append(identifier)
                temperatures.append(temperature)
                pressures.append(pressure)
                break
        else:
            raise ValueError(
                f"Failed to generate unique parameters after {max_attempts} attempts."
            )
        attempts = 0

    # Launch all MD runs in parallel
    for j in range(maxCPUs):
        temperature = temperatures[j]
        pressure = pressures[j]
        identifier = identifiers[j]

        # Set up working directory for this run
        workingFolder = os.path.join(mdFolder, identifier)
        os.mkdir(workingFolder)
        mdFile = os.path.join(workingFolder, identifier + ".in")  # MD input file
        outFile = os.path.join(workingFolder, identifier + ".out")  # MD output file
        timeFile = os.path.join(workingFolder, identifier + ".time")  # Timing file

        # Random lattice parameter variation (±5%)
        latticeParameter = config["baseLatticeParameter"] * random.uniform(0.95, 1.05)

        # Prepare MD simulation properties
        mdProperties = {
            "latticeParameter": latticeParameter,
            "temperature": temperature,
            "pressure": pressure,
            "potFile": potFile,  # Current MTP potential
            "boxDimensions": config["mdLatticeConfigs"][i],  # Simulation cell size
            "elements": config["elements"],  # Chemical elements
            "atomicWeights": config["atomicWeights"],  # Atomic masses
        }

        # Write MD input file and launch simulation
        wr.write_Cu111(mdFile, mdProperties)
        subprocesses.append(
            subprocess.Popen(
                [
                    "/usr/bin/time",  # Track execution time
                    "-o", timeFile,
                    "-f", "%e",  # Output format: elapsed time in seconds
                    "mpirun",
                    "-np", "1",  # Single MPI process
                    "--bind-to", "none",  # No CPU binding
                    config["lmpMPIFile"],  # LAMMPS MPI executable
                    "-in", mdFile,  # Input file
                    "-log", outFile,  # Log file
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workingFolder,  # Run in working directory
            )
        )

    # Wait for all runs to complete and check exit codes
    exitCodes = [p.wait() for p in subprocesses]
    for exitCode in exitCodes:
        if exitCode != 0 and exitCode != 9:  # Allow exit code 9 (user interrupt)
            raise RuntimeError("MD runs have failed!")

    # Process results
    preselectedIterationLogs = {}  # Store grades of selected configurations
    cpuTimesSpent = []  # Track computational cost

    for j in range(maxCPUs):
        identifier = identifiers[j]
        workingFolder = os.path.join(mdFolder, identifier)
        preselectedFile = os.path.join(workingFolder, "preselected.cfg.0")  # Output file

        # Parse execution time
        timeFile = os.path.join(workingFolder, identifier + ".time")
        cpuTimesSpent.append(pa.parseTimeFile(timeFile))

        preselectedGrades = []  # Grades for this run's selected configs

        # If preselected configurations exist, process them
        if os.path.exists(preselectedFile):
            hasPreselected = True
            with open(preselectedFile, "r") as src:
                content = src.read()
                # Extract all grade values using regex
                preselectedGrades = list(
                    map(float, re.findall(r"(?<=MV_grade\t)\d+.?\d*", content))
                )
                # Append to master file
                with open(masterPreselectedFile, "a") as dest:
                    dest.write(content)

        # Store grades keyed by run identifier
        preselectedIterationLogs[identifier] = preselectedGrades

    return (
        exitCodes,
        preselectedIterationLogs,
        hasPreselected,
        cpuTimesSpent,
    )
