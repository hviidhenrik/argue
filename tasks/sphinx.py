import os
import shutil
from invoke import task

import pathlib

_ROOT = pathlib.Path(__file__).parent.parent


@task
def make_html(cmd):
    """Builds the html document from makefile

    Arguments:
        cmd {[type]} -- [description]
    """
    print(
        """
Builds the html document from makefile
===============================================
"""
    )
    cmd.run(r"make html")


@task
def run(cmd):
    """Run sphinx

    Arguments:
        cmd {[type]} -- [description]
    """
    print(
        """
Running Sphinx
===============================================
"""
    )
    cmd.run(r"start docs/build/index.html")


@task
def sync(command):  # pylint: disable=unused-argument
    """Copies the list of files to the docs/source/ folder
    The purpose is to include additional files in the documentation (such as README in root folder location).
    Note that the files will only be synced when you run a sphinx task, not if you only update the files in the project folder and then git commit.
    """
    files = [f"{_ROOT}/README.md"]

    for source_file in files:
        shutil.copy2(source_file, _ROOT / "docs/source")


@task(pre=[sync])
def build(command):
    if os.path.isdir(_ROOT / "docs/build"):
        print("Removing 'build'")
        shutil.rmtree(_ROOT / "docs/build")
    if os.path.isdir(_ROOT / "docs/source/apidoc-generated"):
        print("Removing source/apidoc-generated")
        shutil.rmtree(_ROOT / "docs/source/apidoc-generated")
    command.command_prefixes.insert(0, f"cd {_ROOT}")
    command.run("sphinx-apidoc.exe -feo docs/source/apidoc-generated src -t docs/source/_templates")

    # Adds each .me path here. automate this to find every .md file in the code?
    paths = []

    for path in paths:
        shutil.copyfile(path, _ROOT / "docs\\source" + "\\" + path.split("\\")[-1])

    command.run("sphinx-build.exe docs/source docs/build")
    command.run("start docs/build/index.html")
