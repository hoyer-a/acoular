#%%

from mosqito import loudness_zwst
import pyfar as pf
import numpy as np
import tracemalloc
import time
import logging

# %%
fs = 44.1e3
duration = 60 * 2.5
n_samples = fs* duration
num_channels = 92
signal = pf.signals.noise(n_samples, rms = np.random.random(num_channels))
print(signal.time.T.shape)

# %%
# Configure logging to write to a file, setting the log level and format
logging.basicConfig(filename='mosqito_performance.log', level=logging.INFO, 
                    format='%(message)s')

tracemalloc.start()
start_time = time.time()

n, nspec = loudness_zwst(signal.time.T.squeeze(), fs)[0:2]
current, peak = tracemalloc.get_traced_memory()

end_time = time.time()
elapsed_time = end_time - start_time


# Example log messages
logging.info(f'Numchannels: {num_channels}, signal duration: {duration/60} min,'
             f' memory: {peak/10e6:.2f} MB, time: {elapsed_time/60:.2f} min')

print(f"peak memory use was {peak/10e6} MB")
# %%
