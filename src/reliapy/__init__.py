"""
Reliapy
=======
"""

import pkg_resources

import reliapy.state_limit
import reliapy.transformation
import reliapy.distributions.continuous
import reliapy.monte_carlo
import reliapy._messages
import reliapy.sampling

from reliapy.state_limit import *
from reliapy.transformation import *
from reliapy.distributions.continuous import *
from reliapy.monte_carlo import *
from reliapy._messages import *
from reliapy.sampling import *

try:
    __version__ = pkg_resources.get_distribution("reliapy").version
except pkg_resources.DistributionNotFound:
    __version__ = None

try:
    __version__ = pkg_resources.get_distribution("reliapy").version
except pkg_resources.DistributionNotFound:
    __version__ = None
