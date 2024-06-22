import sys
import os
from os import path

package_path = os.path.abspath('/Users/sebastianlis/Documents/TU/9. Semester/Python : Akkustik/acoular-1')
if package_path not in sys.path:
    sys.path.insert(0, package_path)

import acoular

print(acoular.__file__)

from acoular import (
    BeamformerBase,
    BeamformerCleantSqTraj,
    BeamformerTimeSq,
    BeamformerTimeSqTraj,
    FiltFiltOctave,
    L_p,
    MicGeom,
    Mixer,
    MovingPointSource,
    PointSource,
    PowerSpectra,
    RectGrid,
    SineGenerator,
    SteeringVector,
    TimeAverage,
    TimeCache,
    Trajectory,
    WNoiseGenerator,
    WriteH5
)
from numpy import arange, array, cos, linspace, pi, sin, zeros
from pylab import axis, colorbar, figure, imshow, show, subplot, text, tight_layout, title, transpose

# ===============================================================================
# some important definitions
# ===============================================================================

freq = 2500 # frequency of interest (114 Hz)
sfreq = 48000  # sampling frequency (3072 Hz)
r = 3.0  # array radius
R = 2.5  # radius of source trajectory
Z = 4  # distance of source trajectory from
rps = 15.0 / 60.0  # revolutions per second
U = 3.0  # total number of revolutions
h5savefile = 'rotating_source_2.h5'

# ===============================================================================
# construct the trajectory for the source
# ===============================================================================

tr = Trajectory()
tr1 = Trajectory()
tmax = U / rps
delta_t = 1.0 / rps / 16.0  # 16 steps per revolution
for t in arange(0, tmax * 1.001, delta_t):
    i = t * rps * 2 * pi  # angle
    # define points for trajectory spline
    tr.points[t] = (R * cos(i), R * sin(i), Z)  # anti-clockwise rotation
    tr1.points[t] = (R * cos(i), R * sin(i), Z)  # anti-clockwise rotation

# ===============================================================================
# define circular microphone array
# ===============================================================================

m = MicGeom()
# set 8 microphone positions
m.mpos_tot = array(
    [(r * sin(2 * pi * i + pi / 4), r * cos(2 * pi * i + pi / 4), 0) for i in linspace(0.0, 1.0, 2, False)],
).T

# ===============================================================================
# define the different source signals
# ===============================================================================
if sys.version_info > (3,):
    long = int
nsamples = long(sfreq * tmax)
n1 = WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples)
s1 = SineGenerator(sample_freq=sfreq, numsamples=nsamples, freq=freq)
s2 = SineGenerator(sample_freq=sfreq, numsamples=nsamples, freq=freq, phase=pi)

# ===============================================================================
# define the moving source and one fixed source
# ===============================================================================

p0 = MovingPointSource(signal=s1, mics=m, trajectory=tr1)
# t = p0 # use only moving source
p1 = PointSource(signal=n1, mics=m, loc=(0, R, Z))
t = Mixer(source=p0, sources=[p1])  # mix both signals
# t = p1 # use only fix source

wh5 = WriteH5(source=t, name=h5savefile)
wh5.save()


# uncomment to save the signal to a wave file
# ww = WriteWAV(source = t)
# ww.channels = [0,14]
# ww.save()