"""
Reliapy
=======
"""

import pkg_resources

import reliapy.model
import reliapy.approximative
import reliapy.distributions
import reliapy.monte_carlo

From reliapy.model import *
From reliapy.approximative import *
From reliapy.distributions import *
From reliapy.monte_carlo import *

try:
    __version__ = pkg_resources.get_distribution("reliapy").version
except pkg_resources.DistributionNotFound:
    __version__ = None

try:
    __version__ = pkg_resources.get_distribution("reliapy").version
except pkg_resources.DistributionNotFound:
    __version__ = None
