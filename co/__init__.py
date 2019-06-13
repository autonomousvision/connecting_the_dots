# import os
# this_dir = os.path.dirname(__file__)
# print(this_dir)
# import sys
# sys.path.append(this_dir)

# set matplotlib backend depending on env
import os
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
  matplotlib.use('Agg')

from . import geometry
from . import plt
from . import plt2d
from . import plt3d
from . import metric
from . import table
from . import utils
from . import io3d
from . import gtimer
from . import cmap
from . import args
