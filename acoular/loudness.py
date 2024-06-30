import numpy as np
import warnings
from scipy.signal import resample
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

    bark_axis = CArray(desc="Bark axis in 0.1 bark steps (size = 240)")

    time_axis = CArray(desc="Time axis for timevariant loudness")

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

    @observe('source', post_init=True)
    def _source_changed(self, event):
        """
        Observer method that is called whenever the `source` attribute changes.
        """
        self._calculate_loudness()

    def _calculate_loudness(self):
        """
        Private function to calculate overall and specific loudness.
        """
        print("Calculating stationary loudness... depending on the file size, " 
              "this might take a while")

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
                N, N_spec, self.bark_axis = loudness_zwst(self.source.data[:,i], 
                                          self.sample_freq)[0:3]
                self.overall_loudness[i] = N
                self.specific_loudness[:,i] = N_spec

        else:   
            self.overall_loudness, self.specific_loudness, self.bark_axis = \
                loudness_zwst(self.source.data[:], self.sample_freq)[0:3]
            

class LoudnessTimevariant(_Loudness, HasPrivateTraits):
    """
    Calculates the timevariant loudness according to ISO 532-1 (Zwicker) 
    from a given source. Calculates loudness for timesteps of 64 samples.

    Uses third party code from `mosqito 
    <https://mosqito.readthedocs.io/en/latest/index.html>`__.

    Parameters
    ==========
    source : SamplesGenerator or derived object?
        The input source as TimeSamples object.

    References
    ==========
    - Acoustics –
      Methods for calculating loudness –
      Part 1: Zwicker method (ISO 532-1:2017, Corrected version 2017-11)
    - Green Forge Coop. (2024). MOSQITO (Version 1.2.1) [Computer software]. 
      https://doi.org/10.5281/zenodo.11026796
    """
    overall_loudness = Union(CArray(),
                             desc="overall loudness (shape: `N_channels x" 
                             " N_times)")

    specific_loudness = CArray(desc="specific loudness sones/bark per channel "
                               "(shape: `N_channels x N_bark x N_times`).")

    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @observe('source', post_init=True)
    def _source_changed(self, event):
        """
        Observer method that is called whenever the `source` attribute changes.
        """
        self._calculate_loudness()

    def _calculate_loudness(self):
        """
        Private function to calculate overall and specific loudness.
        """
        print("Calculating timevariant loudness... depending on the file size, " 
              "this might take a while")
        
        # resampling necessary for mosqito tv to work with fs > 48 kHz
        if self.sample_freq > 48000:

            _resampled_data = CArray(desc="resampled data for timevariant "
                                     "loudness")

            self._resampled_data = \
                resample(self.source.data[:], 
                         int(48000 * self.numsamples / self.sample_freq))
            self.sample_freq = 48000
            self.numsamples= int(48000 * self.numsamples / self.sample_freq)

        # get ntime, code from mosqito
        dec_factor = int(self.sample_freq / 2000)
        n_time = int(len(self._resampled_data[:,0][::dec_factor]) / 4)

        self.overall_loudness = np.zeros((self.numchannels, n_time))
        self.specific_loudness = np.zeros((self.numchannels, 240, n_time)) # restructure plot code to bark x channels x time as in stationary loudness?

        for i in range(self.numchannels):
            overall_loudness, specific_loudness, self.bark_axis, self.time_axis\
                = loudness_zwtv(self._resampled_data[:,i], self.sample_freq)

            self.overall_loudness[i,:] = overall_loudness
            self.specific_loudness[i, :, :] = specific_loudness
        

