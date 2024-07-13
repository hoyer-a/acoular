"""Example "Loudness".
acoular example_loudness.py

Demonstrates different features of Loudness class.

Uses measured data in file example_data.h5,
microphone geometry in array_56.xml (part of Acoular).


"""

from os import path

import acoular
import numpy as np
from pylab import colorbar, figure, imshow, show, subplot, tight_layout, title

# ===============================================================================
# example to visualize stationary loudness
# use qt5 backend to enable interactive plot
# ===============================================================================

#datafile = path.join(os.getcwd(),'data\\three_sources.h5')
#micgeofile = path.join(os.getcwd(),'data\\array_64.xml')

#mg = acoular.MicGeom(from_file = micgeofile)
#ts = acoular.MaskedTimeSamples(name = datafile)

#ld_st = acoular.LoudnessStationary(source=ts)
#ld_st.show(mg)

# ===============================================================================
# example to visualize time variant loudeness
# use qt5 backend to enable interactive plot
# ===============================================================================

micgeofile = path.join(path.split(acoular.__file__)[0], 'xml', 'array_56.xml')
datafile = 'example_data.h5'

mg = acoular.MicGeom(from_file=micgeofile)
ts = acoular.TimeSamples(name=datafile)

ld_tv = acoular.LoudnessTimevariant(source=ts)
ld_tv.show(mg)

# ===============================================================================
# example in beamforming application
# ===============================================================================
micgeofile = path.join(path.split(acoular.__file__)[0], 'xml', 'array_56.xml')
datafile = 'example_data.h5'

mg = acoular.MicGeom(from_file=micgeofile)
ts = acoular.MaskedTimeSamples(name=datafile)

#ts.start = 0.2*48000

rg = acoular.RectGrid(x_min=-0.6, x_max=0.0, y_min=-0.3, y_max=0.3, z=0.68, increment=0.05)
env = acoular.Environment(c=346.04)
st = acoular.SteeringVector(grid=rg, mics=mg, env=env)

# ===============================================================================
# delay and sum beamformer in time domain
# processing chain: beamforming, loudness
# ===============================================================================

bt = acoular.BeamformerTime(source=ts, steer=st)

ld_bt = acoular.LoudnessStationary(source=bt, field_type="diffuse")

oal = ld_bt.overall_loudness
oal = oal.reshape(rg.shape)

imshow(oal, vmax=np.max(oal), origin='lower', interpolation='nearest', extent=rg.extend())
title('Loudeness')
colorbar()
show()