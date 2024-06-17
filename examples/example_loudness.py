#%%
from os import path

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
    MicGeom,
    LoudnessStationary,
    Plot
)
from acoular import __file__ as bpath
from pylab import axis, colorbar, figure, imshow, plot, show
import numpy as np
import mosqito
import scipy as sc

#%%
fn = r"/Users/sebastianlis/Documents/TU/9. Semester/Python : Akkustik/acoular-1/examples/three_sources.h5"

micgeofile = path.join(path.split(bpath)[0], 'xml', 'array_64.xml')
mg = MicGeom(from_file=micgeofile)

stat = LoudnessStationary(fn)
plot_instance = Plot(stat, mg)

#%%
fs = stat.fs
channels = stat.num_channels
n = stat.num_samples
specific_loudness = stat.specific_loudness
total_loudness = stat.overall_loudness

