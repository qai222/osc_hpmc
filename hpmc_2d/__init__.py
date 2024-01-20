import os.path
from .polygon import render_plato
from .sim_serial import Sim2DSerial

VERTICES_FOLDER = os.path.dirname(os.path.abspath(__file__))
VERTICES_FILE_TIPSPn = os.path.join(VERTICES_FOLDER, "vertices_tips_pn_2d.npy")

