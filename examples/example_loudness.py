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
    LoudnessStationary,
)
from acoular import __file__ as bpath
from pylab import axis, colorbar, figure, imshow, plot, show
import numpy as np
import mosqito
import scipy as sc

#%%
fn = r"C:\Users\HP\acoular\examples\three_sources.h5"

stat = LoudnessStationary(fn)

#%%
fs = stat.fs
channels = stat.num_channels
n = stat.num_samples
specific_loudness = stat.specific_loudness
total_loudness = stat.overall_loudness

