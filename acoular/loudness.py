import numpy as np
import scipy as sc
from mosqito import (
    loudness_zwtv,
    loudness_zwst)
from acoular import (
    TimeSamples,
)
import wave


class _Loudness:
    """
    Parent class for stationary and timevariant loudness classes

    Parameters
    ----------
    filename : string
        Full path to file.
    cailb : Calib, optional
        Calibration data, instance of Calib class
    """

    def __init__(self, filename, calib=None):
        self.file = filename
        self._calib = calib

    def _get_dimensions(self):
        """
        Private method to determine file dimensions (N_samples, N_channels, fs)
        without reading entire data into memory
        """
        if self.file.endswith(".h5"):
            self.ts = TimeSamples(name=self.file, calib=self._calib)
            self.num_channels = self.ts.numchannels
            self.num_samples = self.ts.numsamples
            self.fs = self.ts.sample_freq
        elif self.file.endswith(".wav"):
            with wave.open(self.file, "rb") as file:
                self.num_channels = file.getnchannels()
                self.num_samples = file.getnframes()
                self.fs = file.getframerate()
        else:
            raise TypeError('input file must be h5 or wave')

    def _load_data(self):
        """
        Private method to read time data from file.
        """
        # Check for file type and load data
        if self.file.endswith(".h5"):
            self.data = np.array(self.ts.data[:])
        elif self.file.endswith(".wav"):
            self.data = sc.io.wavfile.read[1]
        else:
            raise TypeError('input file must be h5 or wave')


class LoudnessStationary(_Loudness):
    """
    Calculates the stationary loudness from h5 and wave files.
    """
    def __init__(self, filename):
        super().__init__(filename)  # Call the parent class's initializer
        self._calculate_loudness()  # Call the loudness calculation method

    @property
    def overall_loudness(self):
        """
        Return overall loudness (shape: `N_channels`).
        """
        return self.N

    @property
    def specific_loudness(self):
        """
        Return specific loudness in sones/bark per channel (shape: `N_bark x
        N_channels`).
        """
        return self.N_specific

    def _calculate_loudness(self):
        """
        Private function to calculate overall and specific loudness.
        """
        # get dimensions of file
        self._get_dimensions()
        # check length of input, large files will be processed in blocks
        if self.num_samples < 960000:
            self._load_data()
            self.N, self.N_specific = \
                loudness_zwst(self.data, self.fs)[0:2]
        else:
            print('call block processing function')
            # call block processing method & calculate loudness in blocks


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
        # check length of input, large files will be processed in blocks
        if self.num_samples < 960000:
            # calculate timevariant loudness
            self.N, self.N_specific, self.bark_axis, self.time_axis = \
                loudness_zwtv(self.data[:, 0], self.fs)
        else:
            print('call block processing function')
