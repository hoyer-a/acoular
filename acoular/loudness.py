import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
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

class Plot:
    """
    Class for plotting loudness data from LoudnessStationary or LoudnessTimevariant instances.

    Parameters
    ----------
    loudness_instance : instance
        Instance of LoudnessStationary or LoudnessTimevariant.
    m : MicGeom
        Instance of the MicGeom class with microphone positions.
    """

    def __init__(self, loudness_instance, m):
        self.loudness_instance = loudness_instance
        self.m = m
        self.N = loudness_instance.overall_loudness
        self.N_specific = loudness_instance.specific_loudness
        self.mpos = m.mpos[:2, :]

    def _create_plot(self):
        """
        Create interactive plot.
        """
        self.fig, (self.ax, self.ax2) = plt.subplots(2, 1)
        self.ax.set_title('Click on point to plot specific loudness')
        self.ax.axis('equal')
        self.ax.set_xlabel('x-Position [cm]')
        self.ax.set_ylabel('y-Position [cm]')
        self.ax.grid(True)
        self.line, = self.ax.plot(self.mpos[0, :], self.mpos[1, :], 'o', picker=True, pickradius=5)

        self.browser = PointBrowser(self)
        self.fig.canvas.mpl_connect('pick_event', self.browser.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.browser.on_press)
        plt.show()

    def plot_specific_loudness(self, dataind):
        """
        Method to plot specific loudness.
        """
        self.ax2.clear()
        self.ax2.plot(self.N_specific[:, dataind])
        self.ax2.set_ylim(0, np.max(self.N_specific) + 1)
        self.ax2.set_title('Specific Loudness')
        self.ax2.set_xlabel('Bark')
        self.ax2.set_ylabel('Sone')
        self.ax2.grid(True)
        overall_loudness = self.N[dataind]
        self.textbox = self.ax2.text(0.95, 0.95, '', transform=self.ax2.transAxes,
                                 verticalalignment='top', horizontalalignment='left',
                                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        self.textbox.set_text(f'Overall Loudness: {overall_loudness:.2f} Sone')
    
        self.fig.canvas.draw()

class PointBrowser:
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower Axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points.
    """

    def __init__(self, plot_instance):
        self.plot_instance = plot_instance
        self.lastind = 0
        self.text = self.plot_instance.ax.text(0.05, 0.95, 'selected: none',
                                               transform=self.plot_instance.ax.transAxes, va='top')
        self.selected, = self.plot_instance.ax.plot([self.plot_instance.mpos[0, 0]], 
                                                     [self.plot_instance.mpos[1, 0]], 'o', 
                                                     ms=12, alpha=0.4, color='yellow', visible=False)

    def on_press(self, event):
        if self.lastind is None or event.key not in ('n', 'p'):
            return
        self.lastind = np.clip(self.lastind + (1 if event.key == 'n' else -1), 0, len(self.plot_instance.mpos[0]) - 1)
        self.update()

    def on_pick(self, event):
        if event.artist != self.plot_instance.line:
            return True

        if not len(event.ind):
            return True

        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        distances = np.hypot(x - self.plot_instance.mpos[0, event.ind], y - self.plot_instance.mpos[1, event.ind])
        self.lastind = event.ind[distances.argmin()]
        self.update()

    def update(self):
        if self.lastind is None:
            return

        dataind = self.lastind
        self.plot_instance.plot_specific_loudness(dataind)
        self.selected.set_visible(True)
        self.selected.set_data([self.plot_instance.mpos[0, dataind]], [self.plot_instance.mpos[1, dataind]])
        self.text.set_text(f'selected: {dataind}')
        self.plot_instance.fig.canvas.draw()

