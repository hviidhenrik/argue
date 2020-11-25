from invoke import task

from src.config import get_project_root


@task
def wheel(command):
    """Creates a wheel of the current code base

    Arguments:
        cmd {[type]} -- [description]
        environment {string} -- A space separated list of files and folders to lint
        type {string} -- which type of transform to deploy (incremental or recalculation)
    """

    print(
        """
Running wheel creation
===============================================
"""
    )
    with command.cd(get_project_root()):
        command.run(r"python setup.py bdist_wheel")
