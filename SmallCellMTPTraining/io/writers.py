from datetime import datetime
import numpy as np
import os

from SmallCellMTPTraining.templates import templates as templates
from SmallCellMTPTraining.templates import properties as props


def checkProperties(propertiesToCheck: list, properties: dict):
    """
    Validates that required properties exist in a dictionary.
    
    Args:
        propertiesToCheck (list): List of required property keys
        properties (dict): Dictionary to validate
        
    Returns:
        str: Comma-separated string of missing properties, or None if all present
    """
    missing = []
    for ele in propertiesToCheck:
        if ele not in properties:
            missing.append(ele)
    if missing:
        return ", ".join(missing)
    return None


def writeQEInput(fileName: str, taskProperties: dict) -> str:
    """
    Generates a Quantum ESPRESSO (QE) input file.
    
    Required properties:
        - atomPositions: List of atomic coordinates
        - atomTypes: List of atom type indices
        - superCell: 3x3 matrix of supercell vectors
        - kPoints: k-point grid dimensions [k1, k2, k3]
        - ecutwfc: Wavefunction cutoff energy
        - ecutrho: Charge density cutoff energy  
        - qeOutDir: Output directory
        - elements: List of element symbols
        - atomicWeights: List of atomic masses
        - pseudopotentials: List of pseudopotential filenames
        - pseudopotentialDirectory: Path to pseudopotentials
    
    Args:
        fileName (str): Output file path
        taskProperties (dict): Dictionary containing all required properties
        
    Returns:
        str: The generated QE input file content
    """
    # Validate required properties
    properties = props.qeProperties
    if checkProperties(properties, taskProperties):
        raise Exception(
            "Properties missing: " + checkProperties(properties, taskProperties)
        )

    numAtoms = len(taskProperties["atomPositions"])
    num_elem = len(taskProperties["elements"])
    
    # Format supercell vectors as strings
    superCellStrings = [str(x)[1:-1] for x in taskProperties["superCell"]]

    # Generate ATOMIC_SPECIES block
    atomic_species_str = ""
    for i in range(num_elem):
        atomic_species_str += "{}  {} {}\n".format(
            taskProperties["elements"][i],
            taskProperties["atomicWeights"][i],
            taskProperties["pseudopotentials"][i],
        )

    # Generate ATOMIC_POSITIONS block
    atomPositionsString = []
    for a in range(numAtoms):
        atom_type_index = taskProperties["atomTypes"][a]
        atomPositionsString.append(
            " {} {} {} {} \n".format(
                taskProperties["elements"][atom_type_index],
                taskProperties["atomPositions"][a][0],
                taskProperties["atomPositions"][a][1],
                taskProperties["atomPositions"][a][2],
            )
        )
    atomPositions = "".join(atomPositionsString)

    # Format the complete QE input using template
    newQEInput = templates.qeInputTemplate.format(
        nat=numAtoms,
        ntyp=num_elem,
        ccc="\n  ".join(superCellStrings),
        k1=taskProperties["kPoints"][0],
        k2=taskProperties["kPoints"][1],
        k3=taskProperties["kPoints"][2],
        out=taskProperties["qeOutDir"],
        ecut=taskProperties["ecutwfc"],
        erho=taskProperties["ecutrho"],
        atomic_species=atomic_species_str,
        aaa=atomPositions,
        pseudopotentialDirectory=taskProperties["pseudopotentialDirectory"],
    )

    # Write to file
    with open(fileName, "w") as f:
        f.write(newQEInput)

    return newQEInput


def writeQEJob(fileName: str, jobProperties: dict):
    """
    Generates a job submission script for QE calculations.
    
    Args:
        fileName (str): Output file path
        jobProperties (dict): Dictionary containing job parameters:
            - ncpus: Number of CPUs
            - jobName: Job name
            - maxDuration: Max runtime
            - runFile: Script to run
            - inFile: QE input file
            - outFile: Output file
            - memPerCpu: Memory per CPU
    """
    properties = props.calcJobProperties
    if checkProperties(properties, jobProperties):
        raise Exception(
            "Properties missing: " + checkProperties(properties, jobProperties)
        )

    newQEJob = templates.qeJobTemplate.format(
        cpus=jobProperties["ncpus"],
        jobName=jobProperties["jobName"],
        time=jobProperties["maxDuration"],
        runFile=jobProperties["runFile"],
        folder=os.path.dirname(jobProperties["inFile"]),
        inFile=os.path.basename(jobProperties["inFile"]),
        outFile=jobProperties["outFile"],
        mem=jobProperties["memPerCpu"],
    )
    with open(fileName, "w") as f:
        f.write(newQEJob)
    return newQEJob


def writeMTPConfigs(filename: str, mtpPropertiesList: list):
    """
    Writes configurations in MTP training format.
    
    Format includes:
        - Supercell information
        - Atom positions and types  
        - Forces
        - Energy
        - Virial stresses
        
    Args:
        filename (str): Output file path
        mtpPropertiesList (list): List of configuration dictionaries
    """
    with open(filename, "w+") as f:
        for mtpProperties in mtpPropertiesList:
            f.write("BEGIN_CFG\n")
            f.write(" Size\n")
            f.write("    " + str(len(mtpProperties["atomIDs"])) + "\n")
            f.write(" Supercell\n")
            for row in mtpProperties["superCell"]:
                f.write("    ")
                f.write("    ".join(map(str, row)))
                f.write("\n")
            f.write(
                " AtomData:  id type    cartes_x    cartes_y    cartes_z    fx    fy    fz\n"
            )
            for atomID, atomType, pos, force in zip(
                mtpProperties["atomIDs"],
                mtpProperties["atomTypes"],
                mtpProperties["atomPositions"],
                mtpProperties["atomForces"],
            ):
                f.write("    ")
                f.write(str(int(atomID)))
                f.write("    ")
                f.write(str(int(atomType)))
                f.write("    ")
                f.write("    ".join(map(str, pos)))
                f.write("    ")
                f.write("    ".join(map(str, force)))
                f.write("\n")
            f.write(" Energy\n    ")
            f.write(str(mtpProperties["energy"]))
            f.write(
                "\n PlusStress:  xx          yy          zz          yz          xz          xy\n     "
            )
            f.write("    ".join(map(str, mtpProperties["virialStresses"])))
            f.write("\n Feature EFS_by Qe\n")
            f.write("END_CFG\n\n")


def writeMDJob(fileName: str, jobProperties: dict):
    """
    Generates a job submission script for MD simulations.
    
    Args:
        fileName (str): Output file path  
        jobProperties (dict): Dictionary containing:
            - ncpus: Number of CPUs
            - jobName: Job name
            - timeFile: File to store timing info
            - maxDuration: Max runtime
            - runFile: Script to run
            - inFile: MD input file
            - outFile: Output file
            - memPerCpu: Memory per CPU
    """
    properties = props.calcJobProperties
    if checkProperties(properties, jobProperties):
        raise Exception(
            "Properties missing: " + checkProperties(properties, jobProperties)
        )

    newMDJob = templates.mdJobTemplate.format(
        cpus=jobProperties["ncpus"],
        jobName=jobProperties["jobName"],
        timeFile=jobProperties["timeFile"],
        time=jobProperties["maxDuration"],
        runFile=jobProperties["runFile"],
        folder=os.path.dirname(jobProperties["inFile"]),
        inFile=os.path.basename(jobProperties["inFile"]),
        outFile=jobProperties["outFile"],
        mem=jobProperties["memPerCpu"],
    )
    with open(fileName, "w") as f:
        f.write(newMDJob)
    return newMDJob


def writeMDInput(fileName: str, jobProperties: dict):
    """
    Generates an MD simulation input file (LAMMPS format).
    
    Required properties:
        - latticeParameter: Base lattice constant
        - boxDimensions: Simulation box dimensions
        - potFile: Path to potential file
        - temperature: Simulation temperature
        - pressure: Simulation pressure
        - elements: List of element symbols
        - atomicWeights: List of atomic masses
        
    Args:
        fileName (str): Output file path
        jobProperties (dict): Dictionary containing all required properties
    """
    properties = props.mdProperties
    if checkProperties(properties, jobProperties):
        raise Exception(
            "Properties missing: " + checkProperties(properties, jobProperties)
        )
        
    num_elem = len(jobProperties["elements"])
    
    # Calculate number of unit cells in each dimension
    nx = int(np.ceil(jobProperties["boxDimensions"][0]))
    ny = int(np.ceil(jobProperties["boxDimensions"][1]))
    nz = int(np.ceil(jobProperties["boxDimensions"][2]))
    
    # Total atoms (2 per unit cell for BCC)
    num_atoms = 2 * nx * ny * nz  

    # Create atom types (random distribution for alloys)
    if num_elem == 1:  # Single element
        atom_types = [0] * num_atoms  
    else:  # Multi-element alloy
        atom_types = []
        for i in range(10):  # Try up to 10 times to get valid distribution
            if len(np.unique(atom_types)) > 1:
                break

            # Generate random concentrations
            concentrations = np.random.uniform(0.01, 1, size=num_elem)
            concentrations /= np.sum(concentrations)  # Normalize

            atom_types = np.random.choice(
                np.arange(num_elem), size=num_atoms, p=concentrations
            )

    # Create mass definitions
    mass_block = ""
    for i in range(num_elem):
        mass_block += "mass  {} {}\n".format(i + 1, jobProperties["atomicWeights"][i])

    # Create atom positions (BCC lattice)
    create_atoms_block = ""
    n_created = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for base_atom in [(0, 0, 0), (0.5, 0.5, 0.5)]:  # BCC basis
                    if n_created < num_atoms:
                        x = (i + base_atom[0]) / nx * jobProperties["boxDimensions"][0]
                        y = (j + base_atom[1]) / ny * jobProperties["boxDimensions"][1]
                        z = (k + base_atom[2]) / nz * jobProperties["boxDimensions"][2]

                        atom_type = atom_types[n_created] + 1  # LAMMPS uses 1-based

                        create_atoms_block += (
                            "create_atoms {} single {} {} {}\n".format(
                                atom_type, x, y, z
                            )
                        )
                        n_created += 1

    # Format complete MD input
    newMDInput = templates.mdInputTemplate.format(
        base=jobProperties["latticeParameter"],
        ttt=jobProperties["temperature"],
        ppp=jobProperties["pressure"],
        pot=jobProperties["potFile"],
        xdim=jobProperties["boxDimensions"][0],
        ydim=jobProperties["boxDimensions"][1],
        zdim=jobProperties["boxDimensions"][2],
        num_elem=num_elem,
        mass_block=mass_block,
        create_atoms_block=create_atoms_block,
    )

    with open(fileName, "w") as f:
        f.write(newMDInput)

    return newMDInput


def writeTrainJob(fileName: str, jobProperties: dict):
    """
    Generates a job script for MTP training.
    
    Args:
        fileName (str): Output file path
        jobProperties (dict): Dictionary containing:
            - ncpus: Number of CPUs
            - jobName: Job name  
            - timeFile: Timing output file
            - maxDuration: Max runtime
            - runFile: Script to run
            - memPerCpu: Memory per CPU
            - potFile: Potential file
            - trainFile: Training set file
            - initRandom: Random initialization flag
            - mode: Training mode
    """
    properties = props.trainJobProperties
    if checkProperties(properties, jobProperties):
        raise Exception(
            "Properties missing: " + checkProperties(properties, jobProperties)
        )
        
    newTrainJob = templates.trainJobTemplate.format(
        cpus=jobProperties["ncpus"],
        jobName=jobProperties["jobName"],
        timeFile=jobProperties["timeFile"],
        time=jobProperties["maxDuration"],
        runFile=jobProperties["runFile"],
        mem=jobProperties["memPerCpu"],
        pot=jobProperties["potFile"],
        train=jobProperties["trainFile"],
        init=jobProperties["initRandom"],
        mode=jobProperties["mode"],
    )

    with open(fileName, "w") as f:
        f.write(newTrainJob)

    return newTrainJob


def writeSelectJob(fileName: str, jobProperties: dict):
    """
    Generates a job script for configuration selection.
    
    Args:
        fileName (str): Output file path
        jobProperties (dict): Dictionary containing:
            - ncpus: Number of CPUs
            - jobName: Job name
            - timeFile: Timing output file  
            - maxDuration: Max runtime
            - runFile: Script to run
            - memPerCpu: Memory per CPU
            - potFile: Potential file
            - trainFile: Training set file
            - preselectedFile: Preselected configs file
            - diffFile: Output diff configs file
    """
    properties = props.selectJobProperties
    if checkProperties(properties, jobProperties):
        raise Exception(
            "Properties missing: " + checkProperties(properties, jobProperties)
        )

    newSelectJob = templates.selectJobTemplate.format(
        cpus=jobProperties["ncpus"],
        jobName=jobProperties["jobName"],
        timeFile=jobProperties["timeFile"],
        time=jobProperties["maxDuration"],
        runFile=jobProperties["runFile"],
        mem=jobProperties["memPerCpu"],
        pot=jobProperties["potFile"],
        train=jobProperties["trainFile"],
        preselected=jobProperties["preselectedFile"],
        diff=jobProperties["diffFile"],
    )
    with open(fileName, "w") as f:
        f.write(newSelectJob)
    return newSelectJob


def printAndLog(fileName: str, message: str):
    """
    Prints a timestamped message and logs it to a file.
    
    Args:
        fileName (str): Log file path
        message (str): Message to log
    """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    datedMessage = dt_string + "   " + message
    print(datedMessage)
    with open(fileName, "a") as myfile:
        myfile.write(datedMessage + "\n")


if __name__ == "__main__":
    # Example usage (you'll need a parsers.py with parseMTPConfigsFile)
    import parsers

    # Create some dummy data for testing
    example_qe_properties = {
        "atomPositions": np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
        "atomTypes": [0, 1],  # Two atoms, first is type 0 (K), second is type 1 (Mg)
        "superCell": np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]]),
        "kPoints": [4, 4, 4],
        "ecutwfc": 30,
        "ecutrho": 120,
        "qeOutDir": "test_output",
        "elements": ["K", "Mg"],
        "atomicWeights": [39.0983, 24.305],
        "pseudopotentials": [
            "K.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "Mg.pbe-spn-kjpaw_psl.1.0.0.UPF",
        ],
    }

    writeQEInput("qe_test_input.in", example_qe_properties)
    example_md_properties = {
        "latticeParameter": 4.2,
        "boxDimensions": [3, 3, 3],
        "potFile": "pot.mtp",
        "temperature": 300,
        "elements": ["K", "Mg"],
        "atomicWeights": [39.0983, 24.305],
    }

    writeMDInput("md_test_input.in", example_md_properties)

    example_md_properties_single = {
        "latticeParameter": 4.2,
        "boxDimensions": [3, 3, 3],
        "potFile": "pot.mtp",
        "temperature": 300,
        "elements": ["K"],
        "atomicWeights": [39.0983],
    }
    writeMDInput("md_test_input_single.in", example_md_properties_single)
