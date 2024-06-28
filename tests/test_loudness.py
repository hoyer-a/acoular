import unittest
import warnings
from pathlib import Path
from acoular import (LoudnessStationary)
from acoular import (
    config,
    TimeSamples,
)
from mosqito import loudness_zwst
import numpy as np

testdir = Path(__file__).parent
moduledir = testdir.parent

config.global_caching = 'none'
datafile = moduledir / 'examples' / 'example_data.h5'
ts = TimeSamples(name=datafile)

class TestStationaryLoudness(unittest.TestCase):
    """
    Test the functionality of the LoudnessStationary class.
    
    Tests only functionality, not the results since LoudnessStationary 
    uses validated third party code"""

    ld_st = LoudnessStationary(source=ts)
    overall_loudness = ld_st.overall_loudness
    specific_loudness = ld_st.specific_loudness

    def test_result_shape(self):
        self.assertEqual(self.overall_loudness.shape[0], ts.numchannels)
        self.assertEqual(self.specific_loudness.shape, (240, ts.numchannels))

    def test_multichanel_handling(self):
        N, N_spec = loudness_zwst(ts.data[:, 1], ts.sample_freq)[0:2]

        np.testing.assert_array_equal(N, self.overall_loudness[1])
        np.testing.assert_array_equal(N_spec, self.specific_loudness[:, 1])

    def test_warnings(self):
        # test for large file warning without time intensive calculation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")
            try:
                ts.numchannels = 97
                ts.numsamples = ts.sample_freq * 3 * 60
                ld_st_ = LoudnessStationary(source=ts)
            except Warning as caught_warning:
                self.skipTest(f"Warning raised: {caught_warning}")            

if __name__ == '__main__':
    unittest.main()

