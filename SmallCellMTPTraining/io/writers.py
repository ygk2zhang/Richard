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
        smearing=taskProperties["smearing"],
        degauss=taskProperties["degauss"],
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

def write_CuHbulk_ase2(fileName: str, jobProperties: dict):
    from ase.build import bulk
    from ase import Atoms
    from ase.io import write
    import numpy as np
    import random
    
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

    a = jobProperties["latticeParameter"]  # Cu lattice constant in Å
    def create_cu_h_supercell(supercell_size=(1,1,1), h_percent=0.25):
        """
           Creates a Cu supercell with H in interstitial sites.

        Args:
        supercell_size (tuple): (nx, ny, nz) replication factors.
        h_percent (float): Fraction of Cu to replace with H (0-0.25).
    
            Returns:
        ase.Atoms: Structure with Cu and H atoms.
        """
        # 1. Create FCC Cu supercell
        a_cu = a  # Å
        cu = bulk('Cu', 'fcc', a=a_cu, cubic=True)
        supercell = cu.repeat(supercell_size)
        
        # 2. Calculate number of Cu atoms to remove
        n_cu = len([a for a in supercell if a.symbol == 'Cu'])
        n_remove = int(n_cu * h_percent)
        if n_remove == 0:
            print("Warning: No Cu atoms removed (h_percent too small for supercell).")
            return supercell
        
        # 3. Remove random Cu atoms
        cu_indices = [i for i, atom in enumerate(supercell) if atom.symbol == 'Cu']
        np.random.shuffle(cu_indices)
        del supercell[cu_indices[:n_remove]]
        
        # 4. Generate all possible interstitial sites
        def generate_interstitial_sites(cell, size):
            """Generate tetrahedral and octahedral sites"""
            sites = []
            # Tetrahedral sites (8 per unit cell)
            tetra = [
                [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
                [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
                [0.75, 0.75, 0.75], [0.25, 0.25, 0.75],
                [0.25, 0.75, 0.25], [0.75, 0.25, 0.25]
            ]
            # Octahedral sites (4 per unit cell)
            octa = [
                [0.5, 0.5, 0.5], [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]
            ]
            
            nx, ny, nz = size
            for i, j, k in np.ndindex(nx, ny, nz):
                for site in tetra + octa:
                    frac_pos = [(site[0] + i)/nx, 
                            (site[1] + j)/ny, 
                            (site[2] + k)/nz]
                    sites.append(np.dot(frac_pos, cell))
            return sites
        
        # 5. Find valid H positions (>1.5 Å from any atom)
        h_sites = generate_interstitial_sites(supercell.get_cell(), supercell_size)
        valid_sites = []
        atom_positions = supercell.get_positions()
        
        for site in h_sites:
            if len(atom_positions) == 0:
                valid_sites.append(site)
            else:
                distances = np.linalg.norm(atom_positions - site, axis=1)
                if np.min(distances) > 1.5:  # Minimum distance threshold
                    valid_sites.append(site)
        
        # 6. Add H atoms (CORRECTED SYNTAX)
        np.random.shuffle(valid_sites)
        for site in valid_sites[:n_remove]:
            supercell += Atoms('H', positions=[site])  # Correct way to add atoms
        
        print(f"Supercell {supercell_size}: Removed {n_remove} Cu, added {min(n_remove, len(valid_sites))} H")
        return supercell
    
    properties = props.mdProperties
    if checkProperties(properties, jobProperties):
        raise Exception("Properties missing: " + checkProperties(properties, jobProperties))
        
    # Create Cu bulk
    nx, ny, nz = [int(np.ceil(d)) for d in jobProperties["boxDimensions"]]
    num_elem = len(jobProperties["elements"])
    
    atoms=create_cu_h_supercell(supercell_size=(nx, ny, nz), h_percent=0.25)
    
    if atoms is not None:
        # Get coordinates and symbols
        symbols = np.array(atoms.get_chemical_symbols())
        positions = atoms.get_positions()
        
        # Create species array (1=Cu, 2=H)
        species = np.ones(len(symbols), dtype=int)
        species[symbols == 'H'] = 2
    
      # Create mass definitions
    mass_block = ""
    for i in range(num_elem):
        mass_block += "mass  {} {}\n".format(i + 1, jobProperties["atomicWeights"][i])

    # Create atom positions (surface)
    positions=atoms.get_positions()
    create_atoms_block = ""
    for i, (atom_type, pos) in enumerate(zip(species, positions)):
        create_atoms_block += f"create_atoms {atom_type} single {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"
    
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
        


def write_Cu_surf_vacancyH(fileName: str, jobProperties: dict):
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
    num_elem = len(jobProperties["elements"])
    layers=4
    from ase.build import fcc111, add_adsorbate
    import random
    import numpy as np
    from ase.io import write
    from scipy.spatial import KDTree

    # Create the Cu surface
    atoms = fcc111('Cu', a=3.63, size=(nx, ny, 4), vacuum=8)
    atoms.center()

    # Calculate 10% of total atoms
    total_atoms = len(atoms)
    n_hydrogens = max(1, int(round(0.1 * total_atoms)))

    # Identify surface atoms
    z_max = max(atoms.get_positions()[:, 2])
    surface_atoms = [a.index for a in atoms if a.z > z_max - 2.0]
    surface_positions = np.array([atoms[i].position[:2] for i in surface_atoms])

    # Create KDTree for efficient nearest-neighbor checks
    kdtree = KDTree(surface_positions)

    # Parameters
    min_h_distance = 1.5  # Minimum distance between H atoms in Å
    max_attempts = 100    # Max attempts to place each H atom

    added_positions = []
    successful_h = 0

    while successful_h < n_hydrogens and len(surface_atoms) > 0:
        attempts = 0
        placed = False
        
        while attempts < max_attempts and not placed:
            # Random vacancy position
            vacancy_pos = np.array([random.uniform(0, atoms.cell[0,0]),
                                random.uniform(0, atoms.cell[1,1])])
            
            # Check distance to existing H atoms
            if len(added_positions) > 0:
                distances = np.linalg.norm(np.array(added_positions) - vacancy_pos, axis=1)
                if np.any(distances < min_h_distance):
                    attempts += 1
                    continue
            
            # Find nearest surface atom
            dist, idx = kdtree.query(vacancy_pos)
            vacancy_index = surface_atoms[idx]
            
            # Verify atom exists and remove it
            if vacancy_index in [a.index for a in atoms]:
                del atoms[vacancy_index]
                # Remove from surface atoms list
                surface_atoms.pop(idx)
                surface_positions = np.delete(surface_positions, idx, axis=0)
                kdtree = KDTree(surface_positions)
                
                # Add hydrogen with randomized parameters
                h_height = random.uniform(0.8, 1.5)  # Wider height range
                add_adsorbate(atoms, 'H', height=h_height, position=vacancy_pos)
                added_positions.append(vacancy_pos)
                successful_h += 1
                placed = True
            else:
                attempts += 1

    
    # Prepare atom types (0 for Cu, 1 for H)
    if atoms is not None:
        # Get coordinates and symbols
        symbols = np.array(atoms.get_chemical_symbols())
        positions = atoms.get_positions()
        
        # Create species array (1=Cu, 2=H)
        species = np.ones(len(symbols), dtype=int)
        species[symbols == 'H'] = 2
    
      # Create mass definitions
    mass_block = ""
    for i in range(num_elem):
        mass_block += "mass  {} {}\n".format(i + 1, jobProperties["atomicWeights"][i])

    # Create atom positions (surface)
    positions=atoms.get_positions()
    create_atoms_block = ""
    for i, (atom_type, pos) in enumerate(zip(species, positions)):
        create_atoms_block += f"create_atoms {atom_type} single {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"
    
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
    atoms.center()
    num_elem = len(jobProperties["elements"])
    # Add hydrogen atom
    h_pos = jobProperties.get("H_position", (1.5, 0.37))  # Default top site
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
