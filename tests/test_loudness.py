import unittest
import warnings
from pathlib import Path
from acoular import (LoudnessStationary, LoudnessTimevariant)
from acoular import (
    config,
    TimeSamples,
)
from mosqito import loudness_zwst
import numpy as np
import copy

testdir = Path(__file__).parent
moduledir = testdir.parent

config.global_caching = 'none'
datafile = moduledir / 'examples' / 'example_data.h5'
ts = TimeSamples(name=datafile)

class TestLoudnessStationary(unittest.TestCase):
    """
    Test the functionality of the LoudnessStationary class.
    
    Tests only functionality, not the results since LoudnessStationary 
    uses validated third party code."""

    ld_st = LoudnessStationary()
    ld_st.source = ts
    overall_loudness = ld_st.overall_loudness
    specific_loudness = ld_st.specific_loudness
    td = ld_st._time_data
    fs = ld_st.sample_freq

    def test_result_shape(self):
        self.assertEqual(self.overall_loudness.shape[0], ts.numchannels)
        self.assertEqual(self.specific_loudness.shape, (240, ts.numchannels))

    def test_multichannel_handling(self):
        N, N_spec = loudness_zwst(self.td[:,1], self.fs)[0:2]

        np.testing.assert_array_equal(N, self.overall_loudness[1])
        np.testing.assert_array_equal(N_spec, self.specific_loudness[:, 1])

    def test_warnings(self):
        pass

        #tbd


# class TestLoudnessTimevariant(unittest.TestCase):       
#     """
#     Test the functionality of the LoudnessTimevariant class.
    
#     Tests only functionality, not the results since LoudnessTimevariant 
#     uses validated third party code"""   

#     ld_tv = LoudnessTimevariant()
#     ld_tv.source = ts
#     _overall_loudness = ld_tv.overall_loudness
#     _specific_loudness = ld_tv.specific_loudness
#     n_samples = ld_tv.numsamples
#     fs = ld_tv.sample_freq

#     def test_result_shape(self):
#         # maybe compare to mosqito result -> resampling necessary for that
#         self.assertEqual(self._overall_loudness.shape, 
#                          (ts.numchannels, 250))
#         self.assertEqual(self._specific_loudness.shape, 
#                          (ts.numchannels, 240, 250))

        

if __name__ == '__main__':
    unittest.main()

