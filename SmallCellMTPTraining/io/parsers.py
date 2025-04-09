import numpy as np
import re
import os

from SmallCellMTPTraining.templates import properties as properties


def extractAllNumbersFromString(sequence: str) -> list:
    """
    Extracts all numerical values from a string using regex.
    
    Args:
        sequence (str): Input string containing numbers
        
    Returns:
        list: List of extracted numbers as floats
    """
    return [float(x) for x in re.findall(r"-?\d\d*\.?\d*", sequence)]


def parseQEOutput(fileName: str, convertToAngRy=True) -> dict:
    """
    Parses a Quantum ESPRESSO output file and extracts key simulation results.
    
    Args:
        fileName (str): Path to QE output file
        convertToAngRy (bool): Whether to convert units to atomic units
        
    Returns:
        dict: Dictionary containing parsed simulation data including:
            - jobComplete: Boolean indicating successful completion
            - energy: Total energy
            - atom positions/forces/types
            - supercell vectors
            - stress tensors
            - pressure
            - CPU time
    """
    # Set unit conversion factors
    energyConversion = properties.energyConversion
    distanceConversion = properties.distanceConversion
    if not convertToAngRy:
        energyConversion = distanceConversion = 1

    with open(fileName, "r") as txtfile:
        content = txtfile.read()

        # Check for job completion
        if re.search(r"stopping...", content):
            print(fileName + " has failed!!")
            return {"jobComplete": False}

        # Extract total energy
        energy = (
            float(re.findall(r"(?:.*!\s*total energy\s*=\s*)(\S+)", content)[0])
            * energyConversion
        )

        # Parse atomic forces
        atomTypeAndForceStrings = re.findall(r".*atom.*type .*\n", content)
        atomTypeAndForceFloats = np.array(
            [extractAllNumbersFromString(x) for x in atomTypeAndForceStrings]
        )
        atomIDs = atomTypeAndForceFloats[:, 0]
        atomTypes = atomTypeAndForceFloats[:, 1].astype(int)
        atomForces = (
            atomTypeAndForceFloats[:, 2:] * energyConversion / distanceConversion
        )

        # Get lattice parameter
        latticeParameter = extractAllNumbersFromString(
            re.findall(r"celldm\(1\)=\s*\d*\.?\d*", content)[0]
        )[1]

        # Parse atomic positions
        positionVectorsStrings = re.findall(
            r"\d\s*[A-Za-z]+\s+tau\(\s+\d+\) = \(\s*-?\d*\.?\d*\s*-?\d*\.?\d*\s*-?\d*\.?\d*  \)",
            content,
        )
        positionVectorsFloats = np.array(
            [extractAllNumbersFromString(x) for x in positionVectorsStrings]
        )[:, 2:]
        positionVectors = positionVectorsFloats * latticeParameter * distanceConversion

        # Parse supercell vectors
        superCellVectorsStrings = re.findall(r".*a\(\d\).*\n", content)
        superCellVectorsFloats = np.array(
            [extractAllNumbersFromString(x) for x in superCellVectorsStrings]
        )
        superCellVectors = (
            superCellVectorsFloats[:, 1:] * latticeParameter * distanceConversion
        )

        # Calculate cell volume
        A2vA3_1 = (
            superCellVectors[1][1] * superCellVectors[2][2]
            - superCellVectors[1][2] * superCellVectors[2][1]
        )
        A2vA3_2 = (
            superCellVectors[1][2] * superCellVectors[2][0]
            - superCellVectors[1][0] * superCellVectors[2][2]
        )
        A2vA3_3 = (
            superCellVectors[1][0] * superCellVectors[2][1]
            - superCellVectors[1][1] * superCellVectors[2][0]
        )
        V = (
            superCellVectors[0][0] * A2vA3_1
            + superCellVectors[0][1] * A2vA3_2
            + superCellVectors[0][2] * A2vA3_3
        )

        # Locate stress tensor in file
        txtfile.seek(0)
        fileLines = txtfile.readlines()
        index = 0
        timeIndex = 0
        for i, line in enumerate(fileLines):
            if re.search(r"total   stress", line):
                index = i
            if re.search(r"     Parallel routines", line):
                timeIndex = i + 2

        # Parse stress tensor
        v1 = np.array(fileLines[index + 1].split(), dtype=float)
        v2 = np.array(fileLines[index + 2].split(), dtype=float)
        v3 = np.array(fileLines[index + 3].split(), dtype=float)
        stressVectors = (
            np.array([v1, v2, v3])[:, :3] * V * energyConversion / distanceConversion**3
        )

        # Format virial stresses
        virialStressFloats = [
            stressVectors[0][0],
            stressVectors[1][1],
            stressVectors[2][2],
            stressVectors[1][2],
            stressVectors[0][2],
            stressVectors[0][1],
        ]

        # Extract pressure
        pressure = extractAllNumbersFromString(
            re.findall(r"P=\s*-?\d\d*.?\d*", content)[0]
        )[0]

        # Parse CPU time with robust pattern matching
        cpuTimeMatchingPattern = r"""
        PWSCF\s*:\s*  # Match "PWSCF:" with optional whitespace
        (?:           # Non-capturing group for days
        (\d+)d\s*   # Match days (optional) 
        )?           
        (?:           # Non-capturing group for hours
        (\d+)h\s*   # Match hours (optional)
        )?            
        (?:           # Non-capturing group for minutes
        (\d+)m\s*   # Match minutes (optional) 
        )?            
        (?:           # Non-capturing group for seconds
        (\d+(?:\.\d+)?)s   # Match seconds with optional decimal
        )?            
        \s*CPU       # Match "CPU" with optional whitespace
        """
        all_matches = re.findall(cpuTimeMatchingPattern, content, re.VERBOSE)

        # Calculate total seconds from parsed time components
        for match in all_matches:
            days, hours, minutes, seconds = match
            total_seconds = (
                (int(days or 0) * 24 * 60 * 60)
                + (int(hours or 0) * 60 * 60)
                + (int(minutes or 0) * 60)
                + float(seconds or 0)
            )

        return {
            "jobComplete": True,
            "energy": energy,
            "atomIDs": atomIDs,
            "atomTypes": atomTypes,
            "atomPositions": positionVectors,
            "atomForces": atomForces,
            "superCell": superCellVectors,
            "stressVectors": stressVectors,
            "virialStresses": virialStressFloats,
            "pressure": pressure,
            "cpuTimeSpent": total_seconds,
        }


def parseMTPConfig(startIndex: int, fileLines: list, convertFromAngRy=True) -> dict:
    """
    Parses a single MTP configuration from file lines.
    
    Args:
        startIndex (int): Starting line index of configuration
        fileLines (list): All lines from the config file
        convertFromAngRy (bool): Whether to convert from atomic units
        
    Returns:
        dict: Parsed configuration data
    """
    # Set unit conversion factors
    energyConversion = 1 / 13.605693012183622  # Ry to eV
    distanceConversion = 1 / 0.5291772105638411  # Bohr to Angstrom
    if not convertFromAngRy:
        energyConversion = distanceConversion = 1

    # Parse basic configuration info
    numAtoms = int(fileLines[startIndex + 2].split()[0])
    energy = float(fileLines[startIndex + 9 + numAtoms].split()[0]) * energyConversion

    # Parse supercell vectors
    v1 = np.array(fileLines[startIndex + 4].split(), dtype=float)
    v2 = np.array(fileLines[startIndex + 5].split(), dtype=float)
    v3 = np.array(fileLines[startIndex + 6].split(), dtype=float)
    superCellVectors = np.array([v1, v2, v3]) * distanceConversion

    # Parse atomic information
    infoArray = np.zeros((numAtoms, 8))
    for j in range(numAtoms):
        infoArray[j] = np.array(fileLines[startIndex + 8 + j].split(), dtype=float)

    atomIDs = infoArray[:, 0]
    atomTypes = infoArray[:, 1].astype(int)
    atomPositions = infoArray[:, 2:5] * distanceConversion
    atomForces = infoArray[:, 5:] * energyConversion / distanceConversion

    # Parse virial stresses
    virialStresses = np.array(
        fileLines[startIndex + 11 + numAtoms].split(), dtype=float
    )

    return {
        "numAtoms": numAtoms,
        "energy": energy,
        "atomIDs": atomIDs,
        "atomTypes": atomTypes,
        "atomPositions": atomPositions,
        "atomForces": atomForces,
        "superCell": superCellVectors,
        "virialStresses": virialStresses,
    }


def parseMTPConfigsFile(filename: str, convertFromAngRy=True) -> list:
    """
    Parses an entire MTP config file containing multiple configurations.
    
    Args:
        filename (str): Path to config file
        convertFromAngRy (bool): Whether to convert from atomic units
        
    Returns:
        list: List of parsed configurations
    """
    properties = []
    with open(filename, "r") as txtfile:
        fileLines = txtfile.readlines()
        # Find all configuration start markers
        indicies = np.where(np.array(fileLines) == "BEGIN_CFG\n")[0]
        for i in indicies:
            properties.append(parseMTPConfig(i, fileLines, convertFromAngRy))
    return properties


def parseAllQEInDirectory(dirName: str, convertFromAngRy=True) -> list:
    """
    Parses all QE output files in a directory.
    
    Args:
        dirName (str): Directory containing QE output files
        convertFromAngRy (bool): Whether to convert from atomic units
        
    Returns:
        list: List of parsed QE outputs
    """
    if not os.path.isdir(dirName):
        raise Exception("The output directory does not exist!")
    qeOutputs = []
    for filename in sorted(os.listdir(dirName)):
        if filename.endswith(".out"):
            output = parseQEOutput(os.path.join(dirName, filename), convertFromAngRy)
            if output["jobComplete"] == False:
                os.remove(os.path.abspath(filename))
            else:
                qeOutputs.append(output)
    return qeOutputs


def parsePartialMTPConfig(startIndex: int, fileLines: list) -> dict:
    """
    Parses a partial MTP configuration (without energy/forces).
    
    Args:
        startIndex (int): Starting line index
        fileLines (list): All lines from file
        
    Returns:
        dict: Parsed partial configuration
    """
    numAtoms = int(fileLines[startIndex + 2].split()[0])

    # Parse supercell
    v1 = np.array(fileLines[startIndex + 4].split(), dtype=float)
    v2 = np.array(fileLines[startIndex + 5].split(), dtype=float)
    v3 = np.array(fileLines[startIndex + 6].split(), dtype=float)
    superCellVectors = np.array([v1, v2, v3])

    # Parse atomic info
    infoArray = np.zeros((numAtoms, 5))
    for j in range(numAtoms):
        infoArray[j] = np.array(fileLines[startIndex + 8 + j].split(), dtype=float)[:5]

    atomIDs = infoArray[:, 0]
    atomTypes = infoArray[:, 1].astype(int)
    atomPositions = infoArray[:, 2:5]

    # Extract MV grade (selection metric)
    mv_grade_line_index = startIndex + 8 + numAtoms
    mv_grade_match = re.search(
        r"Feature\s+MV_grade\s+([\d\.-]+)", fileLines[mv_grade_line_index]
    )
    mv_grade = float(mv_grade_match.group(1)) if mv_grade_match else None

    return {
        "numAtoms": numAtoms,
        "atomIDs": atomIDs,
        "atomTypes": atomTypes,
        "atomPositions": atomPositions,
        "superCell": superCellVectors,
        "MV_grade": mv_grade,
    }


def parsePartialMTPConfigsFile(filename: str) -> list:
    """
    Parses a file containing partial MTP configurations.
    
    Args:
        filename (str): Path to config file
        
    Returns:
        list: List of parsed partial configurations
    """
    properties = []
    with open(filename, "r") as txtfile:
        fileLines = txtfile.readlines()
        indicies = np.where(np.array(fileLines) == "BEGIN_CFG\n")[0]
        for i in indicies:
            properties.append(parsePartialMTPConfig(i, fileLines))
    return properties


def parseTimeFile(filename: str) -> float:
    """
    Parses a time file, handling potential errors.
    
    Args:
        filename (str): Path to time file
        
    Returns:
        float: Parsed time in seconds
        
    Raises:
        RuntimeError: For unexpected status codes
        ValueError: For malformed files
    """
    try:
        with open(filename, "r") as f:
            content = f.read()

        # Check for error status
        error_match = re.match(r"Command exited with non-zero status (\d+)", content)
        if error_match:
            status_code = int(error_match.group(1))
            if status_code not in (0, 9):  # Allow status 0 (success) and 9 (user interrupt)
                raise RuntimeError(
                    f"Command exited with unexpected status: {status_code}"
                )

            # Extract time value after error message
            time_str = content.replace(error_match.group(0), "").strip()
            if not time_str:
                raise ValueError("Time value not found after removing error message.")
            return float(time_str)
        else:
            content = content.strip()
            if not content:
                raise ValueError("File is empty")
            return float(content)

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: '{filename}'")


def parseMDTime(filename: str) -> float:
    """
    Parses MD wall time from output file.
    
    Args:
        filename (str): Path to MD output file
        
    Returns:
        float: Total wall time in seconds
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    pattern = r"Total wall time:\s*(\d+):(\d+):(\d+)"
    for line in lines:
        match = re.search(pattern, line)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            return (hours * 60 * 60) + (minutes * 60) + seconds


if __name__ == "__main__":
    # Example usage
    print(parseQEOutput("1_0_0_0.out"))
