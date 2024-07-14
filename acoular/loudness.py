"""Implements stationary and timevaraint loudness calculation.

.. autosummary::
    :toctree: generated/

    Loudness
    LoudnessStationary
    LoudnessTimevariant

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import matplotlib.axes as mpl_axes
import matplotlib.collections as mpl_collections
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_webagg_core import NavigationToolbar2WebAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from scipy.signal import (
    resample,
)
from traits.api import (
    CArray,
    Float,
    Int,
    String,
    Trait,
    Union,
    PrototypedFrom,
)
from mosqito import (
    loudness_zwtv,
    loudness_zwst,
    )
from acoular import (
    TimeInOut,
    SamplesGenerator,
)

import math

class Loudness(TimeInOut):
    """
    Base class for stationary and timevariant loudness calculation. 
    Get samples from :attr:`source`.
    
    This class has no real functionality on its own and should not be used 
    directly.

    References
    ==========
    - Acoustics –
      Methods for calculating loudness –
      Part 1: Zwicker method (ISO 532-1:2017, Corrected version 2017-11)
    """
    
    #: Data source; :class:`~acoular.tprocess.SamplesGenerator` or derived object.
    source = Trait(SamplesGenerator)

    #: Number of channels in output, as given by :attr:`source`.
    numchannels = PrototypedFrom('source', 'numchannels')

    #: Float representing the sampling frequency of output signal,
    #:  as given by :attr:`source` (default is 48000)
    sample_freq = Float(48000, desc='Sampling frequency of the calculation, default is 48 kHz')

    #: Integer representing the block size for fetching time data over
    #:  result()-method (default is 4096).
    block_size = Int(4096, desc='Block size for fetching time data, default is 4096')

    #: String ({'free', 'diffuse'}) representing the type of soundfield 
    #:  corresponding to ISO 532-1:2017 (default is free).
    field_type = String("free", desc='({`free`, `diffuse`}) Field type, default is `free` ')

    #: CArray representing the bark axis in 0.1 bark steps for visualizing the 
    #:  loudness data.
    bark_axis = CArray(desc='Bark axis in 0.1 bark steps (size = 240)')

    #: CArray representing the time axis for visualizing the loudness data.
    time_axis = CArray(desc='Time axis for timevariant loudness')

    _time_data = CArray(desc='Time data for loudness calculation')

    _n_samples = Int(source.numsamples,desc='Number of samples for loudness calculation')

    start_sample = Int(0, desc='First sample for calculation')

    end_sample = Int(source.numsamples, desc='Last sample for calculation')

    # Private method to resample the signal from source to 48 kHz
    def _resample_to_48kHz(self):
        self._time_data = \
            resample(self._time_data[:], 
                    int(48000 * self.source.numsamples / 
                        self.source.sample_freq))
        self._n_samples = int(48000 * self.source.numsamples / 
                              self.source.sample_freq)
        print("signal resampled to 48 kHz")

class LoudnessStationary(Loudness):
    """
    Calculates the stationary loudness according to ISO 532-1 (Zwicker) 
    from a given source.
    
    Uses third party code from `mosqito 
    <https://mosqito.readthedocs.io/en/latest/index.html>`__.
    
    References
    ==========
    - Green Forge Coop. (2024). MOSQITO (Version 1.2.1) [Computer software]. 
      https://doi.org/10.5281/zenodo.11026796

    """
    #: Union of Float or CArray representing overall loudness for each channel.
    overall_loudness = Union(Float(), CArray(),desc='overall loudness (shape: `N_channels`)')

    #: CArray representing specific loudness in sones per bark per channel.
    specific_loudness = CArray(desc='specific loudness sones/bark per channel (shape: `N_bark x N_channels`).')

    # observe decorator introduces errors and misbehavior e.g. double calculation
    #@observe('source', post_init=False)
    def _source_changed(self):
        """
        Fetches time data via result() in blocks of size `block_size`.
        """
        print("source changed")

         # Ensure block size is smaller than the number of samples.
        if self.source.numsamples < self.block_size:
            raise ValueError(f"Blocksize ({self.block_size}) must be smaller" 
                             " than the number of samples in the source "
                             f"({self.source.numsamples}).")

        # Initialize time data array.
        self._time_data = np.empty((self.source.numsamples, 
                                    self.source.numchannels))
        i = 0

        # Fetch data in blocks and store in time data array.
        for res in self.source.result(self.block_size):
            n_samples = res.shape[0]
            self._time_data[i : i + n_samples] = res
            i += n_samples
  
        # Call calculating method for stationary loudness
        self._calculate_loudness()

    def _calculate_loudness(self):
        """
        Private function to calculate overall and specific loudness.
        """
        print("Calculating stationary loudness... depending on the file size, " 
              "this might take a while")

        # Resample if sample frequency is not 48 kHz.
        if self.source.sample_freq != 48000:
            self._resample_to_48kHz()

        # check input, large files will be processed channel wise
        if (self._n_samples > self.sample_freq * 2.5 * 60 \
            and self.numchannels > 96) \
                or (self._n_samples > self.sample_freq * 14 * 60 \
                    and self._n_samples > 16):

            warnings.warn("File to big to be processed at once. File will be"
                          " processed channel wise", RuntimeWarning)

            # Initialize loudness arrays.
            self.overall_loudness = np.zeros(self.numchannels)
            self.specific_loudness = np.zeros((240, self.numchannels))

            # Process each channel individually.
            for i in range(self.numchannels):
                N, N_spec, self.bark_axis = loudness_zwst(self._time_data[:,i], 
                                          self.sample_freq, 
                                          field_type=self.field_type)[0:3]
                self.overall_loudness[i] = N
                self.specific_loudness[:,i] = N_spec

        else:   
            # Process all data at once.
            self.overall_loudness, self.specific_loudness, self.bark_axis = \
                loudness_zwst(self._time_data[:], self.sample_freq, 
                              field_type=self.field_type)[0:3]

    def show(self, m):
        """
        Create interactive plot to display the overall loudness over time and 
        the specific loudness over time for each microphone. 

        Be aware: If the plot functionality should be used, the microphone 
        positions must be initiated with :class:`~acoular.microphones.MicGeom`

        Parameters
        ----------
        m : Instance Variable
            :class:`~acoular.microphones.MicGeom` Instance Variable that 
            provides the microphone locations.
        """

        # Call Plotclass for Stationary Loudness 
        plt_st = _PlotclassST(self.overall_loudness, self.specific_loudness, 
                              self.bark_axis, m)
        plt_st.plot()


class LoudnessTimevariant(Loudness):
    """
    Calculates the timevariant loudness according to ISO 532-1 (Zwicker) 
    from a given source.
    
    Uses third party code from `mosqito 
    <https://mosqito.readthedocs.io/en/latest/index.html>`__.

    References
    ==========
    - Green Forge Coop. (2024). MOSQITO (Version 1.2.1) [Computer software].
      https://doi.org/10.5281/zenodo.11026796
    """

    #: CArray representing overall loudness for each channel per time step.
    overall_loudness = Union(CArray(), desc='overall loudness (shape: `N_channels x N_times`)')

    #: CArray representing specific loudness in sones per bark per channel per time step.
    specific_loudness = CArray(desc='specific loudness sones/bark per channel (shape: `N_bark x N_channels x N_times`).')

    # observe decorator introduces errors and misbehavior e.g. double calculation
    #@observe('source', post_init=False)
    def _source_changed(self):
        """Fetches time data via result() in blocks of size `block_size`."""
        # Ensure block size is smaller than the number of samples.
        if self.source.numsamples < self.block_size:
            raise ValueError(f"Blocksize ({self.block_size}) must be smaller"
                             " than the number of samples in the source "
                             f"({self.source.numsamples}).")

        print("source changed")

        # Initialize time data array.
        self._time_data = np.empty((self.source.numsamples, self.numchannels))
        i = 0

        # Fetch data in blocks and store in time data array.
        for res in self.source.result(self.block_size):
            n_samples = res.shape[0]
            self._time_data[i : i + n_samples] = res
            i += n_samples

        # Call calculating method for timevaraint loudness
        self._calculate_loudness()

    def _calculate_loudness(self):
        """Private function to calculate overall and specific loudness."""
        print("Calculating timevariant loudness... depending on the file size,"
              " this might take a while")

        # Resample if sample frequency is not 48 kHz.
        if self.source.sample_freq != 48000:
            self._resample_to_48kHz()

        # Determine number of time steps, code from mosqito
        dec_factor = int(self.sample_freq / 2000)
        n_time = math.ceil(len(self._time_data[:,0][::dec_factor]) / 4)
        
        self.overall_loudness = np.zeros((self.numchannels, n_time))
        self.specific_loudness = np.zeros((240, self.numchannels, n_time))

        # Process each channel individually.
        for i in range(self.numchannels):
            overall_loudness, specific_loudness, self.bark_axis, self.time_axis\
                  = loudness_zwtv(self._time_data[:,i],
                                  self.sample_freq,
                                  field_type=self.field_type)

            self.overall_loudness[i,:] = overall_loudness
            self.specific_loudness[:, i, :] = specific_loudness

    def show(self, m):
        """
        Create interactive plot to display the overall loudness over time and
        the specific loudness over time for each microphone.

        Be aware: If the plot functionality should be used, the microphone
        positions must be initiated with :class:`~acoular.microphones.MicGeom`

        Parameters
        ----------
        m : Instance Variable
            :class:`~acoular.microphones.MicGeom` Instance Variable that
            provides the microphone locations.
        """
        # Call Plotclass for timevariant Loudness
        plt_tv = _PlotclassTV(self.overall_loudness, self.specific_loudness,
                              self.bark_axis, self.time_axis, m)
        plt_tv.plot()

class _PlotclassST:
    """Class for plotting static loudness data."""

    def __init__(self, overall_loudness, specific_loudness, bark_axis, m):
        self.overall_loudness = overall_loudness
        self.specific_loudness = specific_loudness
        self.bark_axis = bark_axis
        self.mpos = m.mpos
        self.line2 = None
        self.textbox = None

    def plot(self):
        """
        Create interactive plot to display the overall loudness and specific
        loudness for each microphone.
        """
        # Create figure with two subplots
        self.fig, (self.ax, self.ax2) = plt.subplots(
            2, 1, figsize=(10, 10), gridspec_kw={'width_ratios': [1]}
        )
        self.fig.suptitle(
            'Stationary Loudness for Microphone Array', fontsize=16
        )

        # Configure main scatter plot (microphone array)
        self.ax.set_title(
            'Click on point or use `n` or `p` to plot channelwise specific'
            ' loudness', fontsize=12
        )
        self.ax.axis('equal')
        self.ax.set_xlabel('x-Position [m]')
        self.ax.set_ylabel('y-Position [m]')
        self.ax.grid(True)

        # Scatter plot of microphone positions with overall loudness as color
        scatter = self.ax.scatter(
            self.mpos[0, :], self.mpos[1, :], c=self.overall_loudness,
            cmap='viridis', picker=True, s=50
        )
        self.line = scatter
        # Create a divider for the existing subplot
        divider = make_axes_locatable(self.ax)
        # Append a new axis to the right of 'ax', with 5% width of 'ax'
        cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust pad
        cbar = self.fig.colorbar(scatter, cax=cax)
        cbar.set_label('Overall Loudness (Sone)')

        # Configure second subplot (specific loudness) and leave it empty
        self.ax2.set_title('Channelwise Specific Loudness', fontsize=12)
        self.ax2.set_xlabel('Bark')
        self.ax2.set_ylabel('Sone')
        self.ax2.grid(True)
        # needed for initial setup and positioning 
        self.ax2.set_ylim(0, 1)

        # Initialize PointBrowser for interactive point selection
        self.browser = _PointBrowser(self)
        self.fig.canvas.mpl_connect('pick_event', self.browser.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.browser.on_press)

        # Adjust layout to prevent overlap
        self.fig.tight_layout()

    def _update_plot(self, dataind):
        """
        Generates the specific loudness plot when a point is selected.

        Parameters
        ----------
        dataind : int
            Index of the selected microphone point.
        """
        # Update axis and plot from specific microphone channel
        if self.line2:
            self.line2.set_xdata(self.bark_axis)
            self.line2.set_ydata(self.specific_loudness[:, dataind])
        else:
            self.line2, = self.ax2.plot(
                self.bark_axis, self.specific_loudness[:, dataind]
            )

        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2.set_ylim(0, np.max(self.specific_loudness) + 1)
        # Set the title dynamically based on dataind
        self.ax2.set_title(
            f'Channelwise Specific Loudness (Microphone {dataind})',
            fontsize=12
        )

        # Display overall loudness value as text annotation
        if self.textbox:
            point_overall_loudness = self.overall_loudness[dataind]
            self.textbox.set_text(
                f'Overall Loudness: {point_overall_loudness:.2f} Sone'
            )
        else:
            point_overall_loudness = self.overall_loudness[dataind]
            self.textbox = self.ax2.text(
                0.05, 0.95, f'Overall Loudness: {point_overall_loudness:.2f} Sone',
                transform=self.ax2.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black',
                          facecolor='white')
            )

        # Redraw the figure canvas to reflect updates
        self.fig.canvas.draw()


class _PlotclassTV:
    """
    Class for plotting animated loudness data from LoudnessTimevariant
    instances.
    """

    def __init__(self, overall_loudness, specific_loudness, bark_axis, 
                 time_axis, m):
        self.overall_loudness = overall_loudness
        self.specific_loudness = specific_loudness
        self.bark_axis = bark_axis
        self.time_axis = time_axis
        self.mpos = m.mpos
        self.colorbar = None

    def plot(self):
        """
        Create interactive plot to display the overall loudness over time and
        the specific loudness over time for each microphone.
        """
        # Set up the main figure with a constrained layout and specific size
        self.fig = plt.figure(figsize=(20, 12), constrained_layout=True)
        self.fig.suptitle(
            'Timevaraiant Loudness for Microphone Array', fontsize=16
        )

        # Use gridspec_kw to control height ratios of rows and spacing between
        # subplots
        spec = self.fig.add_gridspec(2, 3, wspace=0.1)

        self.ax = self.fig.add_subplot(spec[0, 0])
        self.ax2 = self.fig.add_subplot(spec[0, 1])
        self.ax3 = self.fig.add_subplot(spec[1, :2])

        # Configure main scatter plot (microphone array)
        self.ax.set_title(
            'Click on point or use `n` or `p` to plot specific loudness'
        )
        self.ax.axis('equal')
        self.ax.set_xlabel('x-Position [m]')
        self.ax.set_ylabel('y-Position [m]')
        self.ax.grid(True)

        # Scatter plot of microphone positions with averaged overall loudness
        # as color
        scatter = self.ax.scatter(
            self.mpos[0, :], self.mpos[1, :], c=self.overall_loudness.mean(
                axis=1
            ), cmap='viridis', picker=True, s=50
        )
        self.line = scatter

        # Create a divider for the existing subplot
        divider = make_axes_locatable(self.ax)

        # Append a new axis to the right of 'ax', with 5% width of 'ax'
        cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust pad

        cbar = self.fig.colorbar(scatter, cax=cax)
        cbar.set_label('Averaged Overall Loudness (Sone)')

        # Initialize PointBrowser for interactive point selection
        self.browser = PointBrowser(self)
        self.fig.canvas.mpl_connect('pick_event', self.browser.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.browser.on_press)

        # Initial setup for second subplot (overall loudness over time)
        self.ax2.set_title('Overall Loudness Over Time')
        self.ax2.set_xlabel('Time [s]')
        self.ax2.set_ylabel('Overall Loudness [Sone]')
        self.ax2.grid(True)

        # Initial setup for third subplot (specific loudness spectrogram)
        self.ax3.set_title('Specific Loudness Spectrogram')
        self.ax3.set_xlabel('Time [s]')
        self.ax3.set_ylabel('Bark')
        self.ax3.grid(True)
        bark_ticks = np.arange(0, 26, 5)
        bark_tick_labels = [str(tick) for tick in bark_ticks]
        self.ax3.set_yticks(bark_ticks * len(self.bark_axis) // 25)
        self.ax3.set_yticklabels(bark_tick_labels)

        plt.show()

    def _update_plot(self, dataind):
        """
        Updates the overall loudness over time plot and the specific loudness
        spectrogram when a point is selected.

        Parameters
        ----------
        dataind : int
            Index of the selected microphone point.
        """
        # Update overall loudness over time plot
        if self.ax2.lines:
            self.ax2.lines[0].set_ydata(self.overall_loudness[dataind, :])
        else:
            self.ax2.plot(self.time_axis, self.overall_loudness[dataind, :])

        self.ax2.relim()
        self.ax2.autoscale_view()

        # Set title for ax2 subplot
        self.ax2.set_title(
            f'Overall Loudness Over Time (Microphone {dataind})'
        )

        # Update specific loudness spectrogram
        if self.ax3.images:
            self.ax3.images[0].set_array(self.specific_loudness[:, dataind, :])
        else:
            cax = self.ax3.imshow(
                self.specific_loudness[:, dataind, :], aspect='auto',
                cmap='viridis', origin='lower'
            )
            self.colorbar = self.fig.colorbar(
                cax, ax=self.ax3, label='Specific Loudness (Sone/Bark)'
            )

        self.ax3.relim()
        self.ax3.autoscale_view()

        # Set title for ax3 subplot
        self.ax3.set_title(
            f'Specific Loudness Spectrogram (Microphone {dataind})'
        )

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
        self.text = self.plot_instance.ax.text(
            0.05, 0.95, 'Selected: none', transform=self.plot_instance.ax.transAxes,
            va='top', fontsize=16
        )

        # Create a plot marker to highlight the selected point, initially
        # invisible.
        self.selected, = self.plot_instance.ax.plot(
            [self.plot_instance.mpos[0, 0]], [self.plot_instance.mpos[1, 0]],
            'o', ms=12, alpha=0.4, color='yellow', visible=False
        )

    def on_press(self, event):
        """
        Handle key press events to navigate through points.

        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            The key press event containing information about which key was
            pressed.
        """
        # Only handle 'n' for next and 'p' for previous keys.
        if self.lastind is None or event.key not in ('n', 'p'):
            return

        # Update the index based on key press and call update method.
        self.lastind = np.clip(
            self.lastind + (1 if event.key == 'n' else -1), 0,
            len(self.plot_instance.mpos[0]) - 1
        )
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
        distances = np.hypot(
            x - self.plot_instance.mpos[0, event.ind],
            y - self.plot_instance.mpos[1, event.ind]
        )
        self.lastind = event.ind[distances.argmin()]
        self.update()
        return None

    def update(self):
        """Update the plot to reflect the newly selected point."""
        # Check if a valid index is selected.
        if self.lastind is None:
            return

        # Update the plots with data from the selected point.
        self.plot_instance._update_plot(self.lastind)

        # Make the selected point marker visible and update its position.
        self.selected.set_visible(True)
        self.selected.set_data(
            [self.plot_instance.mpos[0, self.lastind]],
            [self.plot_instance.mpos[1, self.lastind]]
        )

        # Update the text label to show the selected point index.
        self.text.set_text(f'selected: {self.lastind}')

        # Redraw the figure canvas to reflect updates.
        self.plot_instance.fig.canvas.draw()
