import numpy as np
import scipy as sc
from mosqito import (
    loudness_zwtv,
    loudness_zwst)
from acoular import (
    TimeSamples,
)
import matplotlib.pyplot as plt


class _Loudness:
    """
    Parent class for stationary and timevariant loudness classes"""
    
    def __init__(self, filename):
        self.file = filename

    def _load_data(self):
        """
        Private method to read data from file. For internal use only
        """

        # Check for file type and load data
        if self.file.endswith(".h5"):
            ts = TimeSamples(name=self.file)
            self.num_channels = ts.numchannels
            self.num_samples = ts.numsamples
            self.fs = ts.sample_freq
            self.data = np.array(ts.data[:])
        elif self.file.endswith(".wav"):
            self.fs, self.data = sc.io.wavfile.read
            self.num_channels = self.data.shape[0]
            self.num_samples = self.data.shape[1]
        else:
            raise TypeError('input file must be h5 or wave')

    def _resample_to_48_kHz(self):
        """
        Resamples a NumPy array from the original sampling rate to the target 
        sampling rate.

        Parameters:
        -----------
        data (np.ndarray) : The input array to be resampled.
        original_rate (float): The original sampling rate in Hz.
        target_rate (float): The target sampling rate in Hz.

        Returns:
        ---------
        np.ndarray: The resampled array.
        """
        # Calculate the number of samples in the resampled array
        num_samples = int(len(self.data) * 48000 / self.fs)

        # Perform the resampling
        resampled_data = sc.signal.resample(self.data, num_samples)

        self.num_samples = num_samples
        self.data = resampled_data
        self.fs = 48000
        return self.data


class LoudnessStationary(_Loudness):
    """
    Calculates the stationary loudness from h5 and wave files.

    Parameters
    ----------
    filename : string
        Full path to file.
    """
    def __init__(self, filename):
        super().__init__(filename)  # Call the parent class's initializer
        self._calculate_loudness()  # Call the loudness calculation method

    # probably better as a sperate function outside the class?!
    # Just for Testing
    def plot_loudness_bark(self):
        """
        Plots the loudness over time
        """
        plt.figure()
        plt.plot(self.bark_axis, self.specific_loudness)
        plt.xlabel('bark')
        plt.ylabel('sone')
        plt.show()
        None

    @property
    def overall_loudness(self):
        """
        Return overall loudness
        """
        return self.N

    @property
    def specific_loudness(self):
        """
        Return specific loudness in sones/bark per time sample
        """
        return self.N_specific

    def _calculate_loudness(self):
        """
        Private function to calculate overall and specific loudness. Further
        returns bark-axis for plotting"""

        # load file
        self._load_data()
        # resamplt to 48 kHz since mosqito only works on 48 kHz
        if self.fs != 48e3:
            self.data = self._resample_to_48_kHz()
        # check length of input, large files will be processed in blocks
        if self.num_samples < 960000:
            # calculate stationary loudness
            self.N, self.N_specific, self.bark_axis = \
                loudness_zwst(self.data[:, 0], self.fs)
        else:
            print('call block processing function')


class LoudnessTimevariant(_Loudness):
    """
    Calculates the timevariant loudness from h5 and wave files.

    Parameters
    ----------
    filename : string
        Full path to file.
    """

    def _calculate_loudness(self):
        """
        Private function to calculate overall and specific loudness. Further
        returns bark-axis and time-axis for plotting"""

        # load file
        self._load_data()
        # resamplt to 48 kHz since mosqito only works on 48 kHz
        if self.fs != 48e3:
            self.data = self._resample_to_48_kHz()
        # check length of input, large files will be processed in blocks
        if self.num_samples < 960000:
            # calculate timevariant loudness
            self.N, self.N_specific, self.bark_axis, self.time_axis = \
                loudness_zwtv(self.data[:, 0], self.fs)
        else:
            print('call block processing function')
