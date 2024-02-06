import os
import numpy as np
import regex as re
import subprocess
import math


from SmallCellMTPTraining.templates import templates as templates
from SmallCellMTPTraining.templates import properties as properties
from SmallCellMTPTraining.io import writers as wr
from SmallCellMTPTraining.io import parsers as pa


def generateInitialDataset(inputFolder: str, outputFolder: str, params: dict):
    # Extract the base data sets from the configurations
    DFT1AtomStrains = np.arange(
        params["1AtomDFTStrainRange"][0],
        params["1AtomDFTStrainRange"][1],
        params["1AtomDFTStrainStep"],
    )
    DFT1AtomShears = np.arange(
        params["1AtomDFTShearRange"][0],
        params["1AtomDFTShearRange"][1],
        params["1AtomDFTShearStep"],
    )
    DFT2AtomStrains = np.arange(
        params["2AtomDFTStrainRange"][0],
        params["2AtomDFTStrainRange"][1],
        params["2AtomDFTStrainStep"],
    )

    jobProperties = {
        "ncpus": 1,
        "maxDuration": "0-0:20",
        "memPerCpu": "4G",
    }

    subprocesses = []

    # Generate and submit the 1Atom Strain runs
    for strain in DFT1AtomStrains:
        workingFolder = os.path.join(
            inputFolder, "1AtomDFTstrain" + str(round(strain, 2))
        )
        inputFile = os.path.join(
            workingFolder, "1AtomDFTstrain" + str(round(strain, 2)) + ".in"
        )
        jobFile = os.path.join(
            workingFolder, "1AtomDFTstrain" + str(round(strain, 2)) + ".qsub"
        )
        runFile = os.path.join(
            workingFolder, "1AtomDFTstrain" + str(round(strain, 2)) + ".run"
        )
        outputFile = os.path.join(
            outputFolder, "1AtomDFTstrain" + str(round(strain, 2)) + ".out"
        )

        os.mkdir(workingFolder)

        # Make modifications to the QE input using regex substitutions
        content = re.sub(
            "\$aaa",
            str(
                round(
                    strain
                    * params["baseLatticeParameter"]
                    / 2
                    / properties.distanceConversion,
                    5,
                )
            ),
            templates.atomStrainTemplate,
        )  # substitute lattice vector marker with the lattice vector
        content = re.sub("\$pseudo_dir", params["pseudopotentialDirectory"], content)
        content = re.sub("\$pseudo", params["pseudopotential"], content)
        content = re.sub("\$out", workingFolder, content)

        # Prepare the job file
        jobProperties["jobName"] = "1AtomDFTstrain" + str(round(strain, 2))
        jobProperties["inFile"] = inputFile
        jobProperties["outFile"] = outputFile
        jobProperties["runFile"] = runFile

        # Write the input and the run file
        wr.writeQEJob(jobFile, jobProperties)
        with open(inputFile, "w") as f:
            f.write(content)

        subprocesses.append(subprocess.Popen(["sbatch", jobFile]))

    exitCodes = [p.wait() for p in subprocesses]
    return exitCodes

    # for shear in DFT1AtomShears:
    #     workingFolder = DFT1AtomShearFolder + "/1AtomDFTshear" + str(round(shear, 2))
    #     inputFile = workingFolder + "/1AtomDFTshear" + str(round(shear, 2)) + ".in"
    #     jobFile = workingFolder + "/1AtomDFTshear" + str(round(shear, 2)) + ".qsub"
    #     outputFile = DFToutputFolder + "/1AtomDFTshear" + str(round(shear, 2)) + ".out"

    #     if not os.path.exists(workingFolder):
    #         os.mkdir(workingFolder)

    #     shutil.copyfile(template1AtomShearDFT, inputFile)
    #     shutil.copyfile(templateDFTJob, jobFile)

    #     # Make modifications to the QE input using regex substitutions
    #     with open(inputFile, "r+") as f:
    #         content = f.read()
    #         contentNew = re.sub(
    #             "\$aaa",
    #             str(round(shear * params["baseLatticeParameter"] / 2, 5)),
    #             content,
    #         )  # substitute lattice vector marker with the lattice vector
    #         contentNew = re.sub(
    #             "\$bbb", str(params["baseLatticeParameter"] / 2), contentNew
    #         )  # substitute lattice vector marker with the lattice vector
    #         contentNew = re.sub(
    #             "\$pseudo_dir", params["pseudopotentialDirectory"], contentNew
    #         )
    #         contentNew = re.sub("\$pseudo", params["pseudopotential"], contentNew)
    #         contentNew = re.sub("\$out", workingFolder, contentNew)
    #         f.seek(0)
    #         f.write(contentNew)
    #         f.truncate()

    #     with open(jobFile, "r+") as f:
    #         content = f.read()
    #         contentNew = re.sub("\$job", "Shear" + str(shear), content)
    #         contentNew = re.sub("\$outfile", workingFolder + "/out.run", contentNew)
    #         contentNew = re.sub(
    #             "\$account", params["slurmParam"]["account"], contentNew
    #         )
    #         contentNew = re.sub(
    #             "\$partition", params["slurmParam"]["partition"], contentNew
    #         )
    #         contentNew = re.sub("\$qos", params["slurmParam"]["qos"], contentNew)
    #         contentNew = re.sub("\$cpus", params["dftJobParam"]["cpus"], contentNew)
    #         contentNew = re.sub("\$time", params["dftJobParam"]["time"], contentNew)
    #         contentNew = re.sub("\$folder", os.path.dirname(inputFile), contentNew)
    #         contentNew = re.sub("\$in", os.path.basename(inputFile), contentNew)
    #         contentNew = re.sub("\$out", outputFile, contentNew)
    #         contentNew = re.sub(
    #             "\$mem", str(params["ramUsagePerConfig"][0]), contentNew
    #         )
    #         f.seek(0)
    #         f.write(contentNew)
    #         f.truncate()

    #         if not dryRun:
    #             subprocesses.append(subprocess.Popen(["sbatch", jobFile]))

    # for strain in DFT2AtomStrains:
    #     workingFolder = DFT2AtomStrainFolder + "/2AtomDFTstrain" + str(round(strain, 2))
    #     inputFile = workingFolder + "/2AtomDFTstrain" + str(round(strain, 2)) + ".in"
    #     jobFile = workingFolder + "/2AtomDFTstrain" + str(round(strain, 2)) + ".qsub"
    #     outputFile = (
    #         DFToutputFolder + "/2AtomDFTstrain" + str(round(strain, 2)) + ".out"
    #     )

    #     if not os.path.exists(workingFolder):
    #         os.mkdir(workingFolder)

    #     shutil.copyfile(template2AtomStrainDFT, inputFile)
    #     shutil.copyfile(templateDFTJob, jobFile)

    #     # Make modifications to the QE input using regex substitutions
    #     with open(inputFile, "r+") as f:
    #         content = f.read()
    #         contentNew = re.sub(
    #             "\$aaa", str(round(strain * params["baseLatticeParameter"], 5)), content
    #         )  # substitute lattice vector marker with the lattice vector
    #         contentNew = re.sub(
    #             "\$pseudo_dir", params["pseudopotentialDirectory"], contentNew
    #         )
    #         contentNew = re.sub("\$pseudo", params["pseudopotential"], contentNew)
    #         contentNew = re.sub("\$out", workingFolder, contentNew)
    #         f.seek(0)
    #         f.write(contentNew)
    #         f.truncate()

    #     with open(jobFile, "r+") as f:
    #         content = f.read()
    #         contentNew = re.sub("\$job", "2Strain" + str(strain), content)
    #         contentNew = re.sub("\$outfile", workingFolder + "/out.run", contentNew)
    #         contentNew = re.sub(
    #             "\$account", params["slurmParam"]["account"], contentNew
    #         )
    #         contentNew = re.sub(
    #             "\$partition", params["slurmParam"]["partition"], contentNew
    #         )
    #         contentNew = re.sub("\$qos", params["slurmParam"]["qos"], contentNew)
    #         contentNew = re.sub("\$cpus", params["dftJobParam"]["cpus"], contentNew)
    #         contentNew = re.sub("\$time", params["dftJobParam"]["time"], contentNew)
    #         contentNew = re.sub("\$folder", os.path.dirname(inputFile), contentNew)
    #         contentNew = re.sub("\$in", os.path.basename(inputFile), contentNew)
    #         contentNew = re.sub("\$out", outputFile, contentNew)
    #         contentNew = re.sub(
    #             "\$mem", str(params["ramUsagePerConfig"][0]), contentNew
    #         )
    #         f.seek(0)
    #         f.write(contentNew)
    #         f.truncate()

    #     if not dryRun:
    #         subprocesses.append(subprocess.Popen(["sbatch", jobFile]))

    # exitCodes = [
    #     p.wait() for p in subprocesses
    # ]  # Wait for all the initial generation to finish
    # subprocesses = []
    # failure = bool(sum(exitCodes))
    # if failure:
    #     printAndLog(
    #         str(sum([1 for x in exitCodes if x != 0]))
    #         + " of the inital DFT runs has been unsuccessful. Exiting now..."
    #     )
    #     exit(1)

    # printAndLog("Initial generation of DFT training dataset has completed.")


def calculateDiffConfigs(
    inputFolder: str,
    outputFolder: str,
    diffFile: str,
    attempt: str,
    stage: int,
    iteration: int,
    config: dict,
):
    newConfigs = pa.parsePartialMTPConfigsFile(diffFile)
    cellDimensions = config["mdLatticeConfigs"][stage]
    kPoints = [math.ceil(config["baseKPoints"] / x) for x in cellDimensions]
    subprocesses = []

    for j, newConfig in enumerate(newConfigs):
        identifier = (
            str(attempt) + "_" + str(stage) + "_" + str(iteration) + "_" + str(j)
        )
        workingFolder = os.path.join(inputFolder, identifier)
        os.mkdir(workingFolder)
        qeFile = os.path.join(workingFolder, identifier + ".in")
        runFile = os.path.join(workingFolder, identifier + ".run")
        outFile = os.path.join(outputFolder, identifier + ".out")
        jobFile = os.path.join(workingFolder, identifier + ".qsub")

        qeProperties = {
            "atomPositions": newConfig["atomPositions"],
            "superCell": newConfig["superCell"],
            "ecutrho": config["ecutrho"],
            "ecutwfc": config["ecutwfc"],
            "qeOutDir": workingFolder,
            "kPoints": kPoints,
        }

        jobProperties = {
            "jobName": identifier,
            "ncpus": config["qeCPUsPerConfig"][stage],
            "memPerCpu": config["qeMemPerConfig"][stage],
            "maxDuration": config["qeTimePerConfig"][stage],
            "inFile": qeFile,
            "outFile": outFile,
            "runFile": runFile,
        }

        wr.writeQEInput(qeFile, qeProperties)
        wr.writeQEJob(jobFile, jobProperties)
        subprocesses.append(subprocess.Popen(["sbatch", jobFile]))

    exitCodes = [p.wait() for p in subprocesses]

    cpuTimesSpent = []
    for j, newConfig in enumerate(newConfigs):
        identifier = (
            str(attempt) + "_" + str(stage) + "_" + str(iteration) + "_" + str(j)
        )
        outFile = os.path.join(outputFolder, identifier + ".out")
        qeOutput = pa.parseQEOutput(outFile)
        cpuTimesSpent.append(qeOutput["cpuTimeSpent"])

    return exitCodes, cpuTimesSpent


if __name__ == "__main__":
    import json
    import shutil

    with open("./config.json", "r") as f:
        out = json.load(f)
        # print(out)
        if os.path.exists("./test"):
            shutil.rmtree("./test")
        os.mkdir("./test")
        generateInitialDataset(
            "/global/home/hpc5146/Projects/SmallCellMTPTraining/SmallCellMTPTraining/activeLearning/test",
            "/global/home/hpc5146/Projects/SmallCellMTPTraining/SmallCellMTPTraining/activeLearning/testout",
            out,
        )
