import numpy as np
import warnings
from scipy.signal import resample
from traits.api import (
    CArray,
    Delegate,
    Float,
    HasPrivateTraits,
    Int,
    String,
    Trait,
    Union,
    observe,
)
from mosqito import (
    loudness_zwtv,
    loudness_zwst,
    loudness_zwst_perseg)
from acoular import (
    TimeInOut,
    SamplesGenerator,
)
import math

class _Loudness(TimeInOut):
    """
    Parent class for stationary and timevariant loudness classes
    """
    source = Trait(SamplesGenerator)
    
    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source')

    #: Number of time data channels
    numchannels = Delegate('source')

    numsamples = Delegate('source')

    bark_axis = CArray(desc="Bark axis in 0.1 bark steps (size = 240)")

    time_axis = CArray(desc="Time axis for timevariant loudness")

    _time_data = CArray(desc="Time data for loudness calculation")

    start_sample = Int(0, desc="First sample for calculation")

    end_sample = Int(numsamples, desc="Last sample for calculation")

    _block_size = Int(4096, desc="Block size for fetching time data")

    field_type = String("free", desc="Field type")

    def _resample_to_48kHz(self):
        self._time_data = \
            resample(self._time_data[:], 
                    int(48000 * self.numsamples / self.sample_freq))
        self.sample_freq = 48000
        self.numsamples= int(48000 * self.numsamples / self.sample_freq)
        print("signal resampled to 48 kHz")

class LoudnessStationary(_Loudness):
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
        Fetches time data via result() in blocks of size `_block_size`.
        """
        self._time_data = np.empty((self.numsamples, self.numchannels))
        i = 0

        for res in self.source.result(self._block_size):
            n_samples = res.shape[0]
            self._time_data[i : i + n_samples] = res
            i += n_samples

        self._calculate_loudness()

    def _calculate_loudness(self):
        """
        Private function to calculate overall and specific loudness.
        """
        print("Calculating stationary loudness... depending on the file size, " 
              "this might take a while")
        
        if self.sample_freq != 48000:
            self._resample_to_48kHz()

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
                N, N_spec, self.bark_axis = loudness_zwst(self._time_data[:,i], 
                                          self.sample_freq, 
                                          field_type=self.field_type)[0:3]
                self.overall_loudness[i] = N
                self.specific_loudness[:,i] = N_spec

        else:   
            self.overall_loudness, self.specific_loudness, self.bark_axis = \
                loudness_zwst(self._time_data[:], self.sample_freq, 
                              field_type=self.field_type)[0:3]
            

class LoudnessTimevariant(_Loudness):
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
        Fetches time data via result() in blocks of size `_block_size`.
        """
        self._time_data = np.empty((self.numsamples, self.numchannels))
        i = 0

        for res in self.source.result(self._block_size):
            n_samples = res.shape[0]
            self._time_data[i : i + n_samples] = res
            i += n_samples

        self._calculate_loudness()

    def _calculate_loudness(self):
        """
        Private function to calculate overall and specific loudness.
        """
        print("Calculating timevariant loudness... depending on the file size, " 
              "this might take a while")
        
        # resample to 48 kHz
        if self.sample_freq != 48000:
            self._resample_to_48kHz()

        # get ntime, code from mosqito
        dec_factor = int(self.sample_freq / 2000)
        n_time = int(len(self._time_data[:,0][::dec_factor]) / 4)

        self.overall_loudness = np.zeros((self.numchannels, n_time))
        self.specific_loudness = np.zeros((self.numchannels, 240, n_time)) # restructure plot code to bark x channels x time as in stationary loudness?

        for i in range(self.numchannels):
            overall_loudness, specific_loudness, self.bark_axis, self.time_axis\
                = loudness_zwtv(self._time_data[:,i], self.sample_freq,
                                field_type=self.field_type)

            self.overall_loudness[i,:] = overall_loudness
            self.specific_loudness[i, :, :] = specific_loudness
    