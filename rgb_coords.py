# File should open rgb image, segmented mask and depth mask.
# Using segmented mask fetch the rgb cutout
# make a rouch depth map of the rgb pixels

import numpy as np
import bpy
import bmesh
import math
from mathutils import Matrix, Vector, Euler
