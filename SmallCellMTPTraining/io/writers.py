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
    num_atoms = 4 * nx * ny * nz  

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
                for base_atom in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:  # FCC basis YZ
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



def write_Cu_vacancyH(fileName: str, jobProperties: dict):
    from ase.build import fcc111, add_adsorbate
    from ase.io import write
    import numpy as np
    
    """
    Generates an MD simulation input file with a vacancy on Cu(111) surface
    and a hydrogen atom in the vacancy site.
    
    Required properties:
        - latticeParameter: Base lattice constant
        - boxDimensions: Simulation box dimensions
        - potFile: Path to potential file
        - temperature: Simulation temperature
        - pressure: Simulation pressure
        - elements: List of element symbols (should include 'Cu' and 'H')
        - atomicWeights: List of atomic masses (should include Cu and H masses)
        - vacancy_position: (x,y) position for vacancy (optional, defaults to top site)
        - H_height: Height above surface for hydrogen (optional, default 1.0 Å)
    """
    
    properties = props.mdProperties
    if checkProperties(properties, jobProperties):
        raise Exception("Properties missing: " + checkProperties(properties, jobProperties))
        
    # Create Cu(111) surface
    nx, ny, nz = [int(np.ceil(d)) for d in jobProperties["boxDimensions"]]
    layers=4
    atoms = fcc111('Cu', a=jobProperties["latticeParameter"], size=(nx, ny, layers), vacuum=8)
    atoms.center()
    # Find and remove surface atom to create vacancy
    vacancy_pos = jobProperties.get("vacancy_position", (0, 0))
    z_max = max(atoms.get_positions()[:,2])
    surface_atoms = [a.index for a in atoms if a.z > z_max - 2.0]  # Find surface atoms
    num_elem = len(jobProperties["elements"])
    # Find atom closest to vacancy position
    distances = [np.linalg.norm(a.position[:2] - np.array(vacancy_pos)) for a in atoms if a.index in surface_atoms]
    vacancy_index = surface_atoms[np.argmin(distances)]
    vacancy_site = atoms[vacancy_index].position.copy()
    del atoms[vacancy_index]
    
    # Add hydrogen atom in the vacancy site
    h_height = jobProperties.get("H_height", 1.0)  # Slightly lower for vacancy site
    add_adsorbate(atoms, 'H', height=h_height, position=vacancy_pos)
    
    # Prepare atom types (0 for Cu, 1 for H)
    atom_types = [0] * (len(atoms) - 1) + [1]
    
      # Create mass definitions
    mass_block = ""
    for i in range(num_elem):
        mass_block += "mass  {} {}\n".format(i + 1, jobProperties["atomicWeights"][i])

    # Create atom positions (surface)
    positions=atoms.get_positions()
    create_atoms_block = ""
    for i, pos in enumerate(positions):
        atom_type = atom_types[i] + 1  # LAMMPS uses 1-based
        create_atoms_block += "create_atoms {} single {} {} {}\n".format(atom_type, pos[0], pos[1], pos[2])
    
    # get box dimensions from ASE structure 
    cell = atoms.get_cell()
     
    xdim = cell[0,0]
    ydim = cell[1,1]
    zdim = cell[2,2]
    xydim= cell [1,0]
    # Format complete MD input
    newMDInput = templates.mdInputTemplate.format(
        base=jobProperties["latticeParameter"],
        ttt=jobProperties["temperature"],
        ppp=jobProperties["pressure"],
        pot=jobProperties["potFile"],
        xdim=xdim,
        ydim=ydim,
        zdim=zdim, 
        xydim=xydim, 
        num_elem=num_elem,
        mass_block=mass_block,
        create_atoms_block=create_atoms_block,
    
    with open(fileName, "w") as f:
        f.write(newMDInput)  
        
def write_Cu111_H(fileName: str, jobProperties: dict):
    from ase.build import fcc111, add_adsorbate
    from ase.io import write
    import numpy as np
    
    """
    Generates an MD simulation input file with a hydrogen atom adsorbed on Cu(111) surface.
    
    Required properties:
        - latticeParameter: Base lattice constant
        - boxDimensions: Simulation box dimensions
        - potFile: Path to potential file
        - temperature: Simulation temperature
        - pressure: Simulation pressure
        - elements: List of element symbols (should include 'Cu' and 'H')
        - atomicWeights: List of atomic masses (should include Cu and H masses)
        - H_position: (x,y) position for hydrogen (optional, defaults to top site)
        - H_height: Height above surface for hydrogen (optional, default 1.5 Å)
    """
    
    properties = props.mdProperties
    if checkProperties(properties, jobProperties):
        raise Exception("Properties missing: " + checkProperties(properties, jobProperties))
        
    # Create Cu(111) surface
    nx, ny, nz = [int(np.ceil(d)) for d in jobProperties["boxDimensions"]]
    layers =4
    atoms = fcc111('Cu', a=jobProperties["latticeParameter"], size=(nx, ny, layers), vacuum=8)
    num_elem = len(jobProperties["elements"])
    # Add hydrogen atom
    h_pos = jobProperties.get("H_position", (0, 0))  # Default top site
    h_height = jobProperties.get("H_height", 1.5)    # Default height 1.5 Å
    add_adsorbate(atoms, 'H', height=h_height, position=h_pos)
    
    # Prepare atom types (0 for Cu, 1 for H)
    atom_types = [0] * (len(atoms) - 1) + [1]
    
    # Create mass definitions
    mass_block = ""
    for i in range(num_elem):
        mass_block += "mass  {} {}\n".format(i + 1, jobProperties["atomicWeights"][i])
    
    # Create atom positions
    create_atoms_block = ""
    for i, pos in enumerate(atoms.get_positions()):
        atom_type = atom_types[i] + 1  # LAMMPS uses 1-based
        create_atoms_block += f"create_atoms {atom_type} single {pos[0]} {pos[1]} {pos[2]}\n"
    
    # Get box dimensions
    cell = atoms.get_cell()
    
    # Format MD input
    newMDInput = templates.mdInputTemplate.format(
        base=jobProperties["latticeParameter"],
        ttt=jobProperties["temperature"],
        ppp=jobProperties["pressure"],
        pot=jobProperties["potFile"],
        xdim=cell[0,0],
        ydim=cell[1,1],
        zdim=cell[2,2],
        xydim=cell[1,0],
        num_elem=2,
        mass_block=mass_block,
        create_atoms_block=create_atoms_block,
    )
    
    with open(fileName, "w") as f:
        f.write(newMDInput)

def write_Cu100(fileName: str, jobProperties: dict):
    from ase.build import fcc100
    from ase.io import write
    import numpy as np
    """
    Generates an MD simulation input file (LAMMPS format) with 10 Å vacuum along z-axis.
    
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
    
    # Total atoms (4 per unit cell for FCC) 4 in the size corresponds to four layer meetings
    atoms = fcc100('Cu', a=jobProperties["latticeParameter"], size=(nx, ny, 4), vacuum=6)
    positions = atoms.get_positions()
    num_atoms = len(positions)
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

    # Create atom positions (surface)
    
    create_atoms_block = ""
    for i, pos in enumerate(positions):
        atom_type = atom_types[i] + 1  # LAMMPS uses 1-based
        create_atoms_block += "create_atoms {} single {} {} {}\n".format(atom_type, pos[0], pos[1], pos[2])
    
    # get box dimensions from ASE structure 
    cell = atoms.get_cell()
     
    xdim = cell[0,0]
    ydim = cell[1,1]
    zdim = cell[2,2]
    xydim= cell [1,0]
    # Format complete MD input
    newMDInput = templates.mdInputTemplate.format(
        base=jobProperties["latticeParameter"],
        ttt=jobProperties["temperature"],
        ppp=jobProperties["pressure"],
        pot=jobProperties["potFile"],
        xdim=xdim,
        ydim=ydim,
        zdim=zdim, # Use modified z-dimension with vacuum
        xydim=xydim, 
        num_elem=num_elem,
        mass_block=mass_block,
        create_atoms_block=create_atoms_block,
    )

    with open(fileName, "w") as f:
        f.write(newMDInput)
def write_Cu111(fileName: str, jobProperties: dict):
    from ase.build import fcc111
    from ase.io import write
    import numpy as np
    """
    Generates an MD simulation input file (LAMMPS format) with 10 Å vacuum along z-axis.
    
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
    
    # Calculate number of unit cells in each dimension and desired Cu layer: in this case it is 4
    nx = int(np.ceil(jobProperties["boxDimensions"][0]))
    ny = int(np.ceil(jobProperties["boxDimensions"][1]))
    nz = int(np.ceil(jobProperties["boxDimensions"][2]))
    layer=4 
    # Total atoms (4 per unit cell for FCC) 4 in the size corresponds to four layer meetings
    atoms = fcc111('Cu', a=jobProperties["latticeParameter"], size=(nx, ny, layer), vacuum=6)
    atoms.center()
    positions = atoms.get_positions()
    num_atoms = len(positions)
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

    # Create atom positions (surface)
    
    create_atoms_block = ""
    for i, pos in enumerate(positions):
        atom_type = atom_types[i] + 1  # LAMMPS uses 1-based
        create_atoms_block += "create_atoms {} single {} {} {}\n".format(atom_type, pos[0], pos[1], pos[2])
    
    # get box dimensions from ASE structure 
    cell = atoms.get_cell()
     
    xdim = cell[0,0]
    ydim = cell[1,1]
    zdim = cell[2,2]
    xydim= cell [1,0]
    # Format complete MD input
    newMDInput = templates.mdInputTemplate.format(
        base=jobProperties["latticeParameter"],
        ttt=jobProperties["temperature"],
        ppp=jobProperties["pressure"],
        pot=jobProperties["potFile"],
        xdim=xdim,
        ydim=ydim,
        zdim=zdim, 
        xydim=xydim, 
        num_elem=num_elem,
        mass_block=mass_block,
        create_atoms_block=create_atoms_block,
    )
    with open(fileName, "w") as f:
        f.write(newMDInput)

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
