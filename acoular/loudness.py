import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import matplotlib.axes as mpl_axes
import matplotlib.collections as mpl_collections
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings
from scipy.signal import (
    resample,
)
from traits.api import (
    CArray,
    Delegate,
    Float,
    Int,
    String,
    Trait,
    Union,
    observe,
    Instance,
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
    # Union of Float or CArray representing overall loudness for each channel.
    overall_loudness = Union(Float(), CArray(),
                            desc="overall loudness (shape: `N_channels`)")

    # CArray representing specific loudness in sones per bark per channel.
    specific_loudness = CArray(desc="specific loudness sones/bark per channel "
                                "(shape: `N_bark x N_channels`).")

    # CArray representing the position of microphones in 3D space.
    mpos = CArray(desc="position of microphone "
                  "(`x_position x y_position x z_position`)")
    
    # Instance of Matplotlib figure to hold the entire plot.
    fig = Instance(mpl_figure.Figure)

    # Instance of Matplotlib axes for the main scatter plot (microphone array).
    ax = Instance(mpl_axes.Axes)

    # Instance of Matplotlib axes for the plot displaying specific loudness.
    ax2 = Instance(mpl_axes.Axes)

    # Instance of Matplotlib PathCollection representing the scatter plot points (microphone markers).
    line = Instance(mpl_collections.PathCollection)

    # Instance of Matplotlib Text object used to display information or annotations on plots.
    textbox = Instance(plt.Text)

    # Instance of PointBrowser (custom class) to handle interactive selection of microphone points.
    browser = Instance(object)

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

    def show(self, m):
        """
        Create interactive plot to display the overall loudness and specific loudness for each microphone.

        Parameters
        ----------
        m : object
            class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
        """
        # Assign microphone positions from the input parameter
        self.mpos = m.mpos

        # Create figure with two subplots
        self.fig, (self.ax, self.ax2) = plt.subplots(2, 1)

        # Configure main scatter plot (microphone array)
        self.ax.set_title('Click on point to plot specific loudness')
        self.ax.axis('equal')
        self.ax.set_xlabel('x-Position [m]')
        self.ax.set_ylabel('y-Position [m]')
        self.ax.grid(True)

        # Scatter plot of microphone positions with overall loudness as color
        scatter = self.ax.scatter(self.mpos[0, :], self.mpos[1, :], c=self.overall_loudness, cmap='viridis', picker=True, s=50)
        self.line = scatter
        cbar = self.fig.colorbar(scatter, ax=self.ax)
        cbar.set_label('Overall Loudness (Sone)')

        # Initialize PointBrowser for interactive point selection
        self.browser = PointBrowser(self)
        self.fig.canvas.mpl_connect('pick_event', self.browser.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.browser.on_press)

        plt.show()

    def _update_plot(self, dataind):
        """
        Generates the specific loudness plot when a point is selected.

        Parameters
        ----------
        dataind : int
            Index of the selected microphone point.

        """
        # Clear the subplot for specific loudness
        self.ax2.clear()

        # Plot specific loudness against Bark scale
        self.ax2.plot(self.bark_axis, self.specific_loudness[:, dataind])
        self.ax2.set_ylim(0, np.max(self.specific_loudness) + 1)
        self.ax2.set_title('Specific Loudness')
        self.ax2.set_xlabel('Bark')
        self.ax2.set_ylabel('Sone')
        self.ax2.grid(True)

        # Display overall loudness value as text annotation
        point_overall_loudness = self.overall_loudness[dataind]
        self.textbox = self.ax2.text(0.05, 0.95, '', transform=self.ax2.transAxes,
                                     verticalalignment='top', horizontalalignment='left',
                                     bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        self.textbox.set_text(f'Overall Loudness: {point_overall_loudness:.2f} Sone')

        # Redraw the figure canvas to reflect updates
        self.fig.canvas.draw()

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
                             " N_times`)")

    specific_loudness = CArray(desc="specific loudness sones/bark per channel "
                               "(shape: `N_bark x N_channels x N_times`).")
    
    # CArray: Array to store the position of microphones in 3D space.
    mpos = CArray(desc="position of microphone "
                  "(`x_position x y_position x z_position`)")

    # Instance of Matplotlib Figure to contain the entire plot.
    fig = Instance(mpl_figure.Figure)

    # Instance of Matplotlib Axes for the main scatter plot (microphone array).
    ax = Instance(mpl_axes.Axes)

    # Instance of Matplotlib PathCollection representing scatter plot points (microphone markers).
    line = Instance(mpl_collections.PathCollection)

    # Instance of Matplotlib Axes for plotting overall loudness over time.
    ax2 = Instance(mpl_axes.Axes)

    # Instance of Matplotlib Axes for plotting specific loudness spectrogram.
    ax3 = Instance(mpl_axes.Axes)

    # Instance of Matplotlib Text object used for annotations or displaying information on plots.
    textbox = Instance(plt.Text)

    # Instance of a colorbar associated with specific loudness spectrogram plot.
    colorbar = Instance(object, allow_none=True)

    # Instance of a custom class or object (e.g., PointBrowser) for interactive point selection.
    browser = Instance(object)

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

    def show(self, m):
        """
        Create interactive plot to display the overall loudness over time and the specific loudness over time for each microphone.

        Parameters
        ----------
        m : object
            class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
        """
        # Assign microphone positions from the input parameter
        self.mpos = m.mpos

        # Set up the main figure with a constrained layout and specific size
        self.fig = plt.figure(figsize=(20, 12), constrained_layout=True)
        spec = self.fig.add_gridspec(2, 3)  
        self.ax = self.fig.add_subplot(spec[0, 0])  
        self.ax2 = self.fig.add_subplot(spec[0, 1])  
        self.ax3 = self.fig.add_subplot(spec[1, :2])  

        # Configure main scatter plot (microphone array)
        self.ax.set_title('Click on point to plot specific loudness')
        self.ax.axis('equal') 
        self.ax.set_xlabel('x-Position [m]')
        self.ax.set_ylabel('y-Position [m]')
        self.ax.grid(True)  

        # Scatter plot of microphone positions with averaged overall loudness as color
        scatter = self.ax.scatter(self.mpos[0, :], self.mpos[1, :], c=self.overall_loudness.mean(axis=1), cmap='viridis', picker=True, s=50)
        self.line = scatter  
        cbar = self.fig.colorbar(scatter, ax=self.ax)  
        cbar.set_label('Overall Loudness (Sone)')  

        # Initialize PointBrowser for interactive point selection
        self.browser = PointBrowser(self)  
        self.fig.canvas.mpl_connect('pick_event', self.browser.on_pick)  
        self.fig.canvas.mpl_connect('key_press_event', self.browser.on_press)  

        plt.show()  

    def _update_plot(self, dataind):
        """
        Updates the overall loudness over time plot and the specific loudness spectogram when a point is selected.

        Parameters
        ----------
        dataind : int
            Index of the selected microphone point.

        """
        # Clear the subplot for overall loudness over time and update with new data
        self.ax2.clear()
        self.ax2.plot(self.time_axis, self.overall_loudness[dataind, :])
        self.ax2.set_title('Overall Loudness Over Time')
        self.ax2.set_xlabel('Time [s]')
        self.ax2.set_ylabel('Overall Loudness [Sone]')
        self.ax2.grid(True)  # Enable grid lines

        # Clear the subplot for specific loudness spectrogram and update with new data
        self.ax3.clear()
        cax = self.ax3.imshow(self.specific_loudness[:, dataind, :], aspect='auto', cmap='viridis', origin='lower')
        self.ax3.set_title(f'Specific Loudness Spectrogram (Channel {dataind})')
        self.ax3.set_xlabel('Time [s]')
        self.ax3.set_ylabel('Bark')

        # Manage the color bar associated with the spectrogram plot
        if self.colorbar:
            self.colorbar.update_normal(cax)  
        else:
            self.colorbar = self.fig.colorbar(cax, ax=self.ax3, label='Loudness (Sone/Bark)')  

        # Redraw the figure canvas to reflect updates
        self.fig.canvas.draw()  

class PointBrowser:
    """
    Interactive class for selecting and highlighting points on a plot.
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower Axes. Use the 'n'
    and 'p' keys to browse through the next and previous points.
    """

    def __init__(self, plot_instance):
        """
        Initialize the PointBrowser with the given plot instance.

        Parameters
        ----------
        plot_instance : object
            Instance of the plot that contains the data and the figure.
        """
        self.plot_instance = plot_instance
        self.lastind = 0
        
        # Create a text label in the plot to show the selected point index.
        self.text = self.plot_instance.ax.text(0.05, 0.95, 'selected: none',
                                               transform=self.plot_instance.ax.transAxes, va='top')
        
        # Create a plot marker to highlight the selected point, initially invisible.
        self.selected, = self.plot_instance.ax.plot([self.plot_instance.mpos[0, 0]],
                                                    [self.plot_instance.mpos[1, 0]], 'o',
                                                    ms=12, alpha=0.4, color='yellow', visible=False)

    def on_press(self, event):
        """
        Handle key press events to navigate through points.

        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            The key press event containing information about which key was pressed.
        """
        # Only handle 'n' for next and 'p' for previous keys.
        if self.lastind is None or event.key not in ('n', 'p'):
            return
        
        # Update the index based on key press and call update method.
        self.lastind = np.clip(self.lastind + (1 if event.key == 'n' else -1), 0, len(self.plot_instance.mpos[0]) - 1)
        self.update()

    def on_pick(self, event):
        """
        Handle pick events to select points on the plot.

        Parameters
        ----------
        event : matplotlib.backend_bases.PickEvent
            The pick event containing information about the picked object.
        """
        # Check if the picked object is the scatter plot points.
        if event.artist != self.plot_instance.line:
            return True

        # Check if any indices were picked.
        if not len(event.ind):
            return True

        # Determine the closest picked point to the mouse click position.
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        distances = np.hypot(x - self.plot_instance.mpos[0, event.ind], y - self.plot_instance.mpos[1, event.ind])
        self.lastind = event.ind[distances.argmin()]
        self.update()

    def update(self):
        """
        Update the plot to reflect the newly selected point.
        """
        # Check if a valid index is selected.
        if self.lastind is None:
            return

        # Update the plots with data from the selected point.
        self.plot_instance._update_plot(self.lastind)
        
        # Make the selected point marker visible and update its position.
        self.selected.set_visible(True)
        self.selected.set_data([self.plot_instance.mpos[0, self.lastind]], [self.plot_instance.mpos[1, self.lastind]])
        
        # Update the text label to show the selected point index.
        self.text.set_text(f'selected: {self.lastind}')
        
        # Redraw the figure canvas to reflect updates.
        self.plot_instance.fig.canvas.draw()
