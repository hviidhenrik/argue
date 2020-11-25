"""Module with Invoke test tasks"""

from invoke import task

source_folder = "src"
config_folder = "config"
output_folder = "tests/test-results"
mypy_report = "test-mypy.xml"
pylint_report = "test-pylint.xml"
test_folder = "tests"


@task
def black(commmand, check="no"):
    """ Run black (autoformatter) on all .py files recursively

    Arguments:
        check {string} -- should be 'yes' if only a check should be performed
    """
    print(
        """
    Running Black on code base
    ===============================================
    """
    )
    if check == "yes":
        commmand.run(f"black . --check --config {config_folder}/config.toml", echo=False)
    else:
        commmand.run(f"black . --config {config_folder}/config.toml", echo=False)


@task
def pytest(command, test_files=test_folder, test_results=output_folder, open_results="no"):
    """Runs pytest to identify failing tests

    :param command: [description]
    :type command: [type]
    :param test_files: A space separated list of folders and files to test. (default: {'tests}), defaults to test_folder
    :type test_files: str, optional
    :param test_results: If not None test reports will be generated in the test_results folder, defaults to output_folder
    :type test_results: str, optional
    :param open_results: If True test reports in the 'test_results' folder will be opened in a browser, defaults to False
    :type open_results: bool, optional
    """

    command_string = f"pytest {test_files}" \
        f" --junitxml={output_folder}/test-pytest_results.xml" \
            f" --cov={source_folder} --cov-config={config_folder}/.coveragerc --cov-report=xml --cov-report=html"
    command.run(command_string, echo=True)

    # Open the test coverage report in a browser
    if test_results and open_results == "yes":
        command.run("start htmlcov/index.html")


@task
def mypy(command, files="src/ tests/", stdout="no"):
    """Runs mypy (static type checker) on all .py files recursively

    Arguments:
        command {[type]} -- [description]
    """
    print(
        f"""
    Running mypy for identifying python type errors in {files} 
    ===============================================
    """
    )
    if stdout == "no":
        command_string = f"mypy {files} --config-file {config_folder}/mypy.ini --junit-xml={output_folder}/{mypy_report}"
    else:
        command_string = f"mypy {files} --config-file {config_folder}/mypy.ini"

    command.run(command_string, echo=True)


@task()
def pylint(command, files="src/", stdout="no"):
    """Runs pylint (linter) on all .py files recursively to identify coding errors

    Arguments:
        command {[type]} -- [description]
        files {string} -- A space separated list of files and folders to lint
    """
    print(
        """
Running pylint.
Pylint looks for programming errors, helps enforcing a coding standard,
sniffs for code smells and offers simple refactoring suggestions.
=======================================================================
"""
    )
    if stdout == "yes":
        command_string = f"pylint --rcfile={config_folder}/pylint.rc {files} -j 4"
    else:
        command_string = f"pylint --rcfile={config_folder}/pylint.rc {files} -j 4 --output-format=pylint2junit.JunitReporter 2>&1 > {output_folder}/{pylint_report}"
    command.run(command_string, echo=True)


@task()
def isort(command):
    """Runs isort (import sorter) on all .py files recursively

    Arguments:
        command {[type]} -- [description]
        files {string} -- A space separated list of files and folders to lint
    """
    print(
        """
Running isort the Python code import sorter
===========================================
"""
    )

    command_string = "isort -rc app.py utils/utils.py"
    command.run(command_string, echo=True)
