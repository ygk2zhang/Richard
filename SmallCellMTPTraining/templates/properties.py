# Conversion constants
energyConversion = 13.605693012183622  # eV to Ry
distanceConversion = 0.5291772105638411  # Ang to Bohr

# A list of the minimum necessary properties to create a Quantum Espresso input file
qeProperties = [
    "atomPositions",
    "atomTypes",
    "superCell",
    "kPoints",
    "ecutwfc",
    "ecutrho",
    "qeOutDir",
]

# A list of the minimum necessary properties to write an MTP configuration
mtpProperties = [
    "numAtoms",
    "energy",
    "atomIDs",
    "atomTypes",
    "atomPositions",
    "atomForces",
    "superCell",
    "virialStresses",
]

# A list of the minimum necessary properties to write a job request for QE or MD
calcJobProperties = [
    "jobName",
    "ncpus",
    "runFile",
    "maxDuration",
    "memPerCpu",
    "inFile",
    "outFile",
]

# A list of the minimum necessary properties to write an MD run
mdProperties = ["potFile", "temperature", "pressure"]

trainJobProperties = [
    "jobName",
    "ncpus",
    "runFile",
    "timeFile",
    "maxDuration",
    "memPerCpu",
    "potFile",
    "trainFile",
    "initRandom",
    "mode",
]

selectJobProperties = [
    "jobName",
    "ncpus",
    "runFile",
    "timeFile",
    "maxDuration",
    "memPerCpu",
    "potFile",
    "trainFile",
    "preselectedFile",
    "diffFile",
]
# # A dictionary of a baseline all possible information extracted from either a MTP configuration or a QE output
# configProperties = {
#     "atomPositions": None,
#     "superCell": None,
#     "kPoints": None,
#     "pseudoDir": None,
#     "pseudo": None,
#     "ecutwfc": None,
#     "ecutrho": None,
#     "qeOutdir": None,
#     "numAtoms": None,
#     "energy": None,
#     "atomIDs": None,
#     "atomTypes": None,
#     "atomPositions": None,
#     "atomForces": None,
#     "superCell": None,
#     "virialStresses": None,
#     "runTime": None,
#     "cpusUsed": None,
# }
