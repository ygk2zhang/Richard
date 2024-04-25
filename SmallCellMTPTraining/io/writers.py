from datetime import datetime
import re
import numpy as np
import os

from SmallCellMTPTraining.templates import templates as templates
from SmallCellMTPTraining.templates import properties as props


def checkProperties(propertiesToCheck: int, propeties: dict):
    missing = []
    for ele in propertiesToCheck:
        if not ele in propeties:
            missing.append(ele)
    if missing:
        return ", ".join(missing)
    return None


def writeQEInput(fileName: str, taskProperties: dict) -> str:
    """Creates a QE input, needing
        properties = [
        "atomPositions",
        "superCell",
        "kPoints",
        "ecutwfc",
        "ecutrho",
        "qeOutDir",
    ]
    """
    properties = props.qeProperties
    if checkProperties(properties, taskProperties):
        raise Exception(
            "Properties missing: " + checkProperties(properties, taskProperties)
        )

    numAtoms = len(taskProperties["atomPositions"])
    newQEInput = re.sub(
        r"\$nnn", str(numAtoms), templates.qeInputTemplate
    )  # substitute nat marker with the number of atoms
    superCellStrings = [str(x)[1:-1] for x in taskProperties["superCell"]]
    newQEInput = re.sub(r"\$ccc", "\n  ".join(superCellStrings), newQEInput)
    newQEInput = re.sub(r"\$k1", str(taskProperties["kPoints"][0]), newQEInput)
    newQEInput = re.sub(r"\$k2", str(taskProperties["kPoints"][1]), newQEInput)
    newQEInput = re.sub(r"\$k3", str(taskProperties["kPoints"][2]), newQEInput)
    newQEInput = re.sub(r"\$out", taskProperties["qeOutDir"], newQEInput)
    newQEInput = re.sub(r"\$ecut", str(taskProperties["ecutwfc"]), newQEInput)
    newQEInput = re.sub(r"\$erho", str(taskProperties["ecutrho"]), newQEInput)

    # Generate a series of string representing the list of atoms and positions
    atomPositionsString = []
    for a in np.arange(numAtoms):
        atomPositionsString.append(
            " K %f %f %f \n"
            % (
                taskProperties["atomPositions"][a][0],
                taskProperties["atomPositions"][a][1],
                taskProperties["atomPositions"][a][2],
            )
        )
    atomPositions = " ".join(atomPositionsString)
    newQEInput = re.sub(
        r"\$aaa", atomPositions, newQEInput
    )  # Subsitiute it in for the marker

    with open(fileName, "w") as f:
        f.write(newQEInput)

    return newQEInput


def writeQEJob(fileName: str, jobProperties: dict):
    """Creates a QE Job, needing:
        jobProperties = [
        "jobName",
        "ncpus",
        "runFile",
        "maxDuration",
        "memPerCpu",
        "inFile",
        "outFile",
    ]"""

    properties = props.calcJobProperties
    if checkProperties(properties, jobProperties):
        raise Exception(
            "Properties missing: " + checkProperties(properties, jobProperties)
        )

    # print(jobProperties)

    newQEJob = re.sub(r"\$cpus", str(jobProperties["ncpus"]), templates.qeJobTemplate)
    newQEJob = re.sub(r"\$jobName", jobProperties["jobName"], newQEJob)
    newQEJob = re.sub(r"\$time", jobProperties["maxDuration"], newQEJob)
    newQEJob = re.sub(r"\$runFile", jobProperties["runFile"], newQEJob)
    newQEJob = re.sub(r"\$folder", os.path.dirname(jobProperties["inFile"]), newQEJob)
    newQEJob = re.sub(r"\$inFile", os.path.basename(jobProperties["inFile"]), newQEJob)
    newQEJob = re.sub(r"\$outFile", jobProperties["outFile"], newQEJob)
    newQEJob = re.sub(r"\$mem", jobProperties["memPerCpu"], newQEJob)

    with open(fileName, "w") as f:
        f.write(newQEJob)

    return newQEJob


def writeMTPConfigs(filename: str, mtpPropertiesList: list):
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
    """Creates a MD Job, needing:
        jobProperties = [
        "jobName",
        "ncpus",
        "runFile",
        "maxDuration",
        "memPerCpu",
        "inFile",
        "outFile",
        "timeFile",
    ]"""

    properties = props.calcJobProperties
    if checkProperties(properties, jobProperties):
        raise Exception(
            "Properties missing: " + checkProperties(properties, jobProperties)
        )

    newMDJob = re.sub(r"\$cpus", str(jobProperties["ncpus"]), templates.mdJobTemplate)
    newMDJob = re.sub(r"\$jobName", jobProperties["jobName"], newMDJob)
    newMDJob = re.sub(r"\$timeFile", jobProperties["timeFile"], newMDJob)
    newMDJob = re.sub(r"\$time", jobProperties["maxDuration"], newMDJob)
    newMDJob = re.sub(r"\$runFile", jobProperties["runFile"], newMDJob)
    newMDJob = re.sub(r"\$folder", os.path.dirname(jobProperties["inFile"]), newMDJob)
    newMDJob = re.sub(r"\$inFile", os.path.basename(jobProperties["inFile"]), newMDJob)
    newMDJob = re.sub(r"\$outFile", jobProperties["outFile"], newMDJob)
    newMDJob = re.sub(r"\$mem", jobProperties["memPerCpu"], newMDJob)

    with open(fileName, "w") as f:
        f.write(newMDJob)

    return newMDJob


def writeMDInput(fileName: str, jobProperties: dict):
    """Creates a MD Input, needing:
    mdProperties = ["latticeParameter", "boxDimensions", "potFile", "temperature"]
    """

    properties = props.mdProperties
    if checkProperties(properties, jobProperties):
        raise Exception(
            "Properties missing: " + checkProperties(properties, jobProperties)
        )

    newMDJob = re.sub(
        r"\$base", str(jobProperties["latticeParameter"]), templates.mdInputTemplate
    )
    newMDJob = re.sub(r"\$ttt", str(jobProperties["temperature"]), newMDJob)
    newMDJob = re.sub(r"\$pot", jobProperties["potFile"], newMDJob)
    newMDJob = re.sub(r"\$111", str(jobProperties["boxDimensions"][0]), newMDJob)
    newMDJob = re.sub(r"\$222", str(jobProperties["boxDimensions"][1]), newMDJob)
    newMDJob = re.sub(r"\$333", str(jobProperties["boxDimensions"][2]), newMDJob)

    with open(fileName, "w") as f:
        f.write(newMDJob)

    return newMDJob


def writeTrainJob(fileName: str, jobProperties: dict):
    """Creates a train Job, needing:
        trainJobProperties = [
        "jobName",
        "ncpus",
        "runFile",
        "maxDuration",
        "memPerCpu",
        "potFile",
        "trainFile",
        "initRandom",
    ]
    """

    properties = props.trainJobProperties
    if checkProperties(properties, jobProperties):
        raise Exception(
            "Properties missing: " + checkProperties(properties, jobProperties)
        )

    newTrainJob = re.sub(
        r"\$cpus", str(jobProperties["ncpus"]), templates.trainJobTemplate
    )
    newTrainJob = re.sub(r"\$jobName", jobProperties["jobName"], newTrainJob)
    newTrainJob = re.sub(r"\$timeFile", jobProperties["timeFile"], newTrainJob)
    newTrainJob = re.sub(r"\$time", jobProperties["maxDuration"], newTrainJob)
    newTrainJob = re.sub(r"\$runFile", jobProperties["runFile"], newTrainJob)
    newTrainJob = re.sub(r"\$mem", jobProperties["memPerCpu"], newTrainJob)
    newTrainJob = re.sub(r"\$pot", jobProperties["potFile"], newTrainJob)
    newTrainJob = re.sub(r"\$train", jobProperties["trainFile"], newTrainJob)
    newTrainJob = re.sub(r"\$init", jobProperties["initRandom"], newTrainJob)

    with open(fileName, "w") as f:
        f.write(newTrainJob)

    return newTrainJob


def writeSelectJob(fileName: str, jobProperties: dict):
    """Creates a select Job, needing:
    selectJobProperties = [
        "jobName",
        "ncpus",
        "runFile",
        "maxDuration",
        "memPerCpu",
        "potFile",
        "trainFile",
        "preselectedFile",
        "diffFile",
        "timeFile",
    ]
    """

    properties = props.selectJobProperties
    if checkProperties(properties, jobProperties):
        raise Exception(
            "Properties missing: " + checkProperties(properties, jobProperties)
        )

    newSelectJob = re.sub(
        r"\$cpus", str(jobProperties["ncpus"]), templates.selectJobTemplate
    )
    newSelectJob = re.sub(r"\$jobName", jobProperties["jobName"], newSelectJob)
    newSelectJob = re.sub(r"\$timeFile", jobProperties["timeFile"], newSelectJob)
    newSelectJob = re.sub(r"\$time", jobProperties["maxDuration"], newSelectJob)
    newSelectJob = re.sub(r"\$runFile", jobProperties["runFile"], newSelectJob)
    newSelectJob = re.sub(r"\$mem", jobProperties["memPerCpu"], newSelectJob)
    newSelectJob = re.sub(r"\$pot", jobProperties["potFile"], newSelectJob)
    newSelectJob = re.sub(r"\$train", jobProperties["trainFile"], newSelectJob)
    newSelectJob = re.sub(
        r"\$preselected", jobProperties["preselectedFile"], newSelectJob
    )
    newSelectJob = re.sub(r"\$diff", jobProperties["diffFile"], newSelectJob)

    with open(fileName, "w") as f:
        f.write(newSelectJob)

    return newSelectJob


def printAndLog(fileName: str, message: str):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    datedMessage = dt_string + "   " + message
    print(datedMessage)
    with open(fileName, "a") as myfile:
        myfile.write(datedMessage + "\n")


if __name__ == "__main__":
    import parsers

    MTPproperties = parsers.parseMTPConfigsFile(
        "/global/home/hpc5146/Projects/K-MTP-training/newPots/16/train.cfg", False
    )
    for i, ele in enumerate(MTPproperties):
        MTPproperties[i]["energy"] -= -0.95127868 * 13.605703976 * len(ele["atomIDs"])
    writeMTPConfigs(
        "/global/home/hpc5146/Projects/K-MTP-training/newPots/16/normalized.cfg",
        MTPproperties,
    )
