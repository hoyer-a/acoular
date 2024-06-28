import numpy as np
import warnings
from traits.api import (
    CArray,
    Delegate,
    Float,
    HasPrivateTraits,
    Int,
    Trait,
    Union,
    observe,
)
from mosqito import (
    loudness_zwtv,
    loudness_zwst,
    loudness_zwst_perseg)
from acoular import (
    TimeSamples,
    TimeInOut,
)
import math

class _Loudness(TimeInOut, HasPrivateTraits):
    """
    Parent class for stationary and timevariant loudness classes
    """
    source = Trait(TimeSamples)
    
    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source')

    #: Number of time data channels
    numchannels = Delegate('source')

    numsamples = Delegate('source')

class LoudnessStationary(_Loudness, HasPrivateTraits):
    """
    Calculates the stationary loudness according to ISO 532-1 (Zwicker) 
    from a given source.

    Uses third party code from `mosqito 
    <https://mosqito.readthedocs.io/en/latest/index.html>`__.

    Parameters
    ==========
    source : TimeSamples
        The input source as TimeSamples object.

    References
    ==========
    - Acoustics –
      Methods for calculating loudness –
      Part 1: Zwicker method (ISO 532-1:2017, Corrected version 2017-11)
    - Mosqito [...] tbd
    """

    overall_loudness = Union(Float(), CArray(),
                             desc="overall loudness (shape: `N_channels`)")

    specific_loudness = CArray(desc="specific loudness sones/bark per channel "
                               "(shape: `N_bark x N_channels`).")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calculate_loudness()

    def _calculate_loudness(self):
        """
        Private function to calculate overall and specific loudness.
        """
        print("Calculating loudness... depending on the file size, this might" 
              " take a while")

        # check input, large files will be processed channel wise
        if (self.numsamples > self.sample_freq * 2.5 * 60 \
            and self.numchannels > 96) \
                or (self.numsamples > self.sample_freq * 14 * 60 \
                    and self.numchannels > 16):

            warnings.warn("File to big to be processed at once. File will be"
                          " processed channel wise", RuntimeWarning)
        
            self.overall_loudness = np.zeros(self.numchannels)
            self.specific_loudness = np.zeros((240, self.numchannels))

            for i in range(self.numchannels):
                N, N_spec = loudness_zwst(self.source.data[:,i], 
                                          self.sample_freq)[0:2]
                self.overall_loudness[i] = N
                self.specific_loudness[:,i] = N_spec

        else:   
            self.overall_loudness, self.specific_loudness = \
                loudness_zwst(self.source.data[:], self.sample_freq)[0:2]
            

