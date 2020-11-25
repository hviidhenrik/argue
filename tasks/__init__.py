"""Here we import the different task submodules/ collections"""
import os

from invoke import Collection

from . import sphinx, test, build

# pylint: disable=invalid-name
# as invoke only recognizes lower case
namespace = Collection()
namespace.add_collection(test)
namespace.add_collection(sphinx)
namespace.add_collection(build)
