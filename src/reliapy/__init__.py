"""
Reliapy
=======
"""

import pkg_resources

import reliapy.state_limit
import reliapy.transfornation
import reliapy.distributions
import reliapy.monte_carlo

from reliapy.state_limit import *
from reliapy.transfornation import *
from reliapy.distributions import *
from reliapy.monte_carlo import *

try:
    __version__ = pkg_resources.get_distribution("reliapy").version
except pkg_resources.DistributionNotFound:
    __version__ = None

try:
    __version__ = pkg_resources.get_distribution("reliapy").version
except pkg_resources.DistributionNotFound:
    __version__ = None
