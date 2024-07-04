import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
from scipy.signal import (
    resample,
    spectrogram,
)
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

    @observe('source', post_init=False)
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

    @observe('source', post_init=False)
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
        self.specific_loudness = np.zeros((240, self.numchannels, n_time))

        for i in range(self.numchannels):
            overall_loudness, specific_loudness, self.bark_axis, self.time_axis\
                = loudness_zwtv(self._time_data[:,i], self.sample_freq,
                                field_type=self.field_type)

            self.overall_loudness[i,:] = overall_loudness
            self.specific_loudness[:, i, :] = specific_loudness
            

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
    