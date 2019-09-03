"""
Execute Notebooks
"""

import os
import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

SOURCE = "_build/website/jupyter/_downloads"
TARGET = "_build/website/jupyter/_downloads/executed"
TIMEOUT = 1300

if not os.path.exists(SOURCE):
    print("Source Directory {} is not found".format(SOURCE))

if not os.path.exists(TARGET):
    print("Setting up execution folder ...")
    os.makedirs(TARGET)

FILES = [fn for fn in glob.glob(SOURCE+"/**/*.ipynb", recursive=True) if not "executed" in fn]
for FL in FILES:
    if "index" in FL:
        print("Skipping: {}".format(FL))
        continue
    if "zreferences" in FL:
        print("Skipping: {}".format(FL))
        continue
    print("Executing: {}".format(FL))
    notebook_name = FL.split("/")[-1]
    #Open Source Notebook
    with open(FL, encoding="UTF-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=TIMEOUT, allow_errors=True)
    out = ep.preprocess(nb, {"metadata": {"path": TARGET}})
    executed_notebook_path = FL.replace(SOURCE, TARGET)
    if not os.path.exists(os.path.dirname(executed_notebook_path)):
        os.makedirs(os.path.dirname(executed_notebook_path))
    print("Writing Executed Notebook: {}".format(executed_notebook_path))
    #Write Executed Notebook as File
    with open(executed_notebook_path, "wt", encoding="UTF-8") as f:
        nbformat.write(nb, f)