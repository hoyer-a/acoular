import unittest
from pathlib import Path
from acoular import (LoudnessStationary, LoudnessTimevariant)
from acoular import (
    config,
    TimeSamples,
    MaskedTimeSamples
)
from mosqito import loudness_zwst, loudness_zwtv
from scipy.signal import resample
import numpy as np

testdir = Path(__file__).parent
moduledir = testdir.parent

config.global_caching = 'none'
datafile = moduledir / 'examples' / 'example_data.h5'
datafile48 = moduledir / 'examples' / 'three_sources_48kHz.h5'
ts = TimeSamples(name=datafile)
ts48 = TimeSamples(name=datafile48)

class TestLoudnessStationary(unittest.TestCase):
    """
    Test the functionality of the LoudnessStationary class.
    
    Tests only functionality, not the results since LoudnessStationary 
    uses validated third party code."""

    # calculation with LoudnessStationary class
    ld_st = LoudnessStationary()
    ld_st.source = ts
    overall_loudness = ld_st.overall_loudness
    specific_loudness = ld_st.specific_loudness
    td = ld_st._time_data
    fs = ld_st.sample_freq

    # Test shapes of overall & specific loudness
    def test_result_shape(self):
        self.assertEqual(self.overall_loudness.shape[0], ts.numchannels)
        self.assertEqual(self.specific_loudness.shape, (240, ts.numchannels))

    # Test shapes of overall & specific loudness
    def test_results(self):
        N, N_spec = loudness_zwst(self.td[:,1], self.fs)[0:2]

        np.testing.assert_array_equal(N, self.overall_loudness[1])
        np.testing.assert_array_equal(N_spec, self.specific_loudness[:, 1])

    def test_errors(self):
        # Test error if Blocksize is bigger than numsamples
        self.ld_st.block_size = self.ld_st.numsamples + 1
        
        with self.assertRaises(ValueError, msg=
                               f"Blocksize ({self.ld_st.block_size}) must be "
                               f"smaller than the number of samples in the "
                               f"source ({self.ld_st.numsamples})."):
            self.ld_st._source_changed()


class TestLoudnessTimevariant(unittest.TestCase):       
    """
    Test the functionality of the LoudnessTimevariant class.
    
    Tests only functionality, not the results since LoudnessTimevariant 
    uses validated third party code"""   
    
    # calculation with LoudnessTimevariant class
    ld_tv = LoudnessTimevariant()
    ld_tv.source = ts48
    _overall_loudness = ld_tv.overall_loudness
    _specific_loudness = ld_tv.specific_loudness
    fs = ld_tv.sample_freq

    # reference calculation with mosqito
    N, N_spec, bark_axis, time_axis = loudness_zwtv(ld_tv._time_data[:,1], 
                                                    fs)

    # Test shapes of overall & specific loudness
    def test_result_shape(self):
        self.assertEqual(self._overall_loudness.shape, 
                         (ts48.numchannels, len(self.time_axis)))
        self.assertEqual(self._specific_loudness.shape, 
                         (len(self.bark_axis), 
                          ts48.numchannels, len(self.time_axis)))
        
    # Test if the results are equal to mosqito results
    def test_compare_result(self):

        np.testing.assert_array_equal(self.N, 
                                      self._overall_loudness[1,:])
        np.testing.assert_array_equal(self.N_spec, 
                                      self._specific_loudness[:, 1, :])

if __name__ == '__main__':
    unittest.main()