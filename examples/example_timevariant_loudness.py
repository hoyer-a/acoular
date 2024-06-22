#%%
import sys
import os
from os import path

package_path = os.path.abspath('/Users/sebastianlis/Documents/TU/9. Semester/Python : Akkustik/acoular-1')
if package_path not in sys.path:
    sys.path.insert(0, package_path)

from acoular import (
    BeamformerBase,
    L_p,
    MicGeom,
    Mixer,
    PointSource,
    PowerSpectra,
    RectGrid,
    SteeringVector,
    TimeSamples,
    TimeInOut,
    WNoiseGenerator,
    WriteH5,
    LoudnessStationary,
    LoudnessTimevariant,
    AnimatedPlot
)
from acoular import __file__ as bpath
from pylab import axis, colorbar, figure, imshow, plot, show
import numpy as np
import mosqito
import scipy as sc

fn = r"/Users/sebastianlis/Documents/TU/9. Semester/Python : Akkustik/acoular-1/examples/rotating_source_2.h5"

m = MicGeom()
# set 28 microphone positions
r = 3.0  # array radius
m.mpos_tot = np.array(
    [(r * np.sin(2 * np.pi * i + np.pi / 4), r * np.cos(2 * np.pi * i + np.pi / 4), 0) for i in np.linspace(0.0, 1.0, 2, False)],
).T

stat = LoudnessTimevariant(fn)

fs = stat.fs
channels = stat.num_channels
n = stat.num_samples
specific_loudness = stat.specific_loudness
total_loudness = stat.overall_loudness

animated_plot = AnimatedPlot(stat, m)
#animated_plot._create_plot()
#plot_instance._create_plot()


# export PYTHONPATH="/Users/sebastianlis/Documents/TU/9. Semester/Python : Akkustik/acoular-1:$PYTHONPATH" source ~/.zshrc  