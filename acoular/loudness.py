import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
            self.N, self.N_specific, self.bark_axis = \
                loudness_zwst(self.data, self.fs)
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
        Private function to calculate overall and specific loudness. Further
        returns bark-axis and time-axis for plotting"""
        # get dimensions of file
        self._get_dimensions()
        # load file
        self._load_data()
        # check length of input, large files will be processed in blocks
        if self.num_samples < 960000:
            # Initialize storage for each variable
            N_list = []
            N_specific_list = []
            bark_axis_list = []
            time_axis_list = []
            for i in range(self.num_channels):
                # calculate timevariant loudness
                N, N_specific, bark_axis, time_axis = \
                    loudness_zwtv(self.data[:,i], self.fs)
                # Append the results to the corresponding list
                N_list.append(N)
                N_specific_list.append(N_specific)
                bark_axis_list.append(bark_axis)
                time_axis_list.append(time_axis)
            # Convert lists to numpy arrays and add a new dimension to separate iterations
            self.N = np.array(N_list)  
            self.N_specific = np.array(N_specific_list)  
            self.bark_axis = np.array(bark_axis_list)  
            self.time_axis = np.array(time_axis_list)  
        else:
            print('call block processing function')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LoudnessMicrophonePlot:
    """
    Base class for plotting loudness data over a microphone array.

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
        self.mpos = m.mpos[:2, :]
        self.N = loudness_instance.overall_loudness
        self.N_specific = loudness_instance.specific_loudness
        self.bark_axis = loudness_instance.bark_axis
        

    def _create_plot(self):
        """
        Abstract method to create the main plot. To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method!")

class PointBrowser:
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower Axes. Use the 'n'
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

        self.plot_instance.update_plot(self.lastind)
        self.selected.set_visible(True)
        self.selected.set_data([self.plot_instance.mpos[0, self.lastind]], [self.plot_instance.mpos[1, self.lastind]])
        self.text.set_text(f'selected: {self.lastind}')
        self.plot_instance.fig.canvas.draw()

class AnimatedPlot(LoudnessMicrophonePlot):
    """
    Class for plotting animated loudness data from LoudnessTimevariant instances.
    """
    def __init__(self, loudness_instance, m):
        super().__init__(loudness_instance, m)
        self.time_steps = self.N.shape[1]  # Assuming the second dimension is time
        self.current_animation = None
        self._create_plot()

    def _create_plot(self):
        """
        Create interactive plot with animation.
        """
        # Set up figure with three subplots
        self.fig = plt.figure(figsize=(20, 12), layout="constrained")
        spec = self.fig.add_gridspec(2, 3)
        self.ax = self.fig.add_subplot(spec[0, :2])
        self.ax2 = self.fig.add_subplot(spec[1, 0])
        self.ax3 = self.fig.add_subplot(spec[1, 1])
        self.ax4 = self.fig.add_subplot(spec[0, 2])

        # Create first subplot displaying the microphone array
        self.ax.set_title('Click on point to plot specific loudness')
        self.ax.axis('equal')
        self.ax.set_xlabel('x-Position [m]')
        self.ax.set_ylabel('y-Position [m]')
        self.ax.grid(True)

        # Use scatter to plot microphones with averaged overall loudness as colorbar
        self.scatter = self.ax.scatter(self.mpos[0, :], self.mpos[1, :], c=self.N.mean(axis=1), cmap='viridis', picker=True, s=50)
        self.line = self.scatter
        self.cbar = self.fig.colorbar(self.scatter, ax=self.ax)
        self.cbar.set_label('Overall Loudness (Sone)')

        # Set up the selection of points to display loudness data
        self.browser = PointBrowser(self)
        self.fig.canvas.mpl_connect('pick_event', self.browser.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.browser.on_press)

        plt.show()

    def plot_N_over_time(self, dataind):
        """
        Update the overall loudness over time plot.
        """
        # Create second subplot displaying the overall loudness of the selected microphone over time 
        self.ax2.clear()
        self.ax2.set_title('Overall Loudness Over Time')
        self.ax2.set_xlabel('Time [s]')
        self.ax2.set_ylabel('Overall Loudness [Sone]')
        self.ax2.grid(True)
        self.ax2.plot(np.linspace(0, self.time_steps / self.loudness_instance.fs, self.time_steps),
                      self.N[dataind, :])
        self.fig.canvas.draw()

    def plot_N_specific_over_time(self, dataind):
        """
        Method to plot and animate specific loudness over the time.
        """
        # Create third subplot displaying the specific loudness of the selected microphone over time
        self.ax3.clear()
        self.ax3.set_ylim(0, np.max(self.N_specific) + 1)
        self.ax3.set_title('Specific Loudness')
        self.ax3.set_xlabel('Bark')
        self.ax3.set_ylabel('Sone')
        self.ax3.grid(True)

        # initialise plot for animation
        self.line2, = self.ax3.plot(self.bark_axis[0], self.N_specific[dataind, :, 0])

        def update(frame):
            self.line2.set_ydata(self.N_specific[dataind, :, frame])
            return self.line2,

        if self.current_animation is not None:
            self.current_animation.event_source.stop()

        self.current_animation = FuncAnimation(self.fig, update, frames=self.time_steps, interval=200, blit=True)

        self.fig.canvas.draw()

    def plot_spectrogram(self, dataind):
        """
        Method to plot the spectrogram of the selected microphone.
        """
        # Create the fourth subplot displaying the spectrogram
        self.ax4.clear()
        self.ax4.set_title('Spectrogram')
        self.ax4.set_xlabel('Time [s]')
        self.ax4.set_ylabel('Frequency [Hz]')

        # Generate the spectrogram
        f, t, Sxx = spectrogram(self.N[dataind, :], self.loudness_instance.fs)
        self.ax4.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        self.ax4.set_ylim([0, self.loudness_instance.fs / 2])
        self.fig.canvas.draw()


    def update_plot(self, dataind):
        """
        Method called by PointBrowser to update plots when a point is selected.
        """
        self.plot_N_over_time(dataind)
        self.plot_N_specific_over_time(dataind)



class StaticPlot(LoudnessMicrophonePlot):
    """
    Class for plotting static loudness data.
    """

    def __init__(self, loudness_instance, m):
        super().__init__(loudness_instance, m)

    def _create_plot(self):
        """
        Create interactive plot.
        """
        self.fig, (self.ax, self.ax2) = plt.subplots(2, 1)
        self.ax.set_title('Click on point to plot specific loudness')
        self.ax.axis('equal')
        self.ax.set_xlabel('x-Position [m]')
        self.ax.set_ylabel('y-Position [m]')
        self.ax.grid(True)

        if self.N.ndim == 2:  # Assuming shape (num_mics, num_time_steps)
            self.N = self.N[:, 0]  # Use the first time step for the scatter plot

        # Use scatter to plot microphones with overall loudness as color
        scatter = self.ax.scatter(self.mpos[0, :], self.mpos[1, :], c=self.N, cmap='viridis', picker=True, s=50)
        self.line = scatter

        # Add color bar
        cbar = self.fig.colorbar(scatter, ax=self.ax)
        cbar.set_label('Overall Loudness (Sone)')

        self.browser = PointBrowser(self)
        self.fig.canvas.mpl_connect('pick_event', self.browser.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.browser.on_press)

        plt.show()

    def plot_specific_loudness(self, dataind):
        """
        Method to plot specific loudness.
        """
        self.ax2.clear()
        self.ax2.plot(self.bark_axis, self.N_specific[:, dataind])
        self.ax2.set_ylim(0, np.max(self.N_specific) + 1)
        self.ax2.set_title('Specific Loudness')
        self.ax2.set_xlabel('Bark')
        self.ax2.set_ylabel('Sone')
        self.ax2.grid(True)
        overall_loudness = self.N[dataind]
        self.textbox = self.ax2.text(0.05, 0.95, '', transform=self.ax2.transAxes,
                                     verticalalignment='top', horizontalalignment='left',
                                     bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        self.textbox.set_text(f'Overall Loudness: {overall_loudness:.2f} Sone')

        self.fig.canvas.draw()

    def update_plot(self, dataind):
        """
        Method called by PointBrowser to update plots when a point is selected.
        """
        self.plot_specific_loudness(dataind)
