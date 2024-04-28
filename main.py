import os
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

from dasIT.data.loader import RFDataloader, TDloader, TGCloader
from dasIT.features.transducer import transducer
from dasIT.features.medium import medium
from dasIT.features.tgc import tg_compensation
from dasIT.src.delays import planewave_delays
from dasIT.src.apodization import apodization
from dasIT.src.das_bf import RXbeamformer
from dasIT.features.signal import RFfilter, fftsignal, analytic_signal
#from dasIT.features.image import interp_lateral
from dasIT.visualization.signal_callback import amp_freq_1channel, transducer_channel_map, IQsignal_1ch
from dasIT.visualization.image_callback import plot_signal_grid, plot_signal_image

####################################################################
#-------------------- Preset MANUAL / USERINPUT -------------------#
ARG_BASE_PATH = pathlib.PureWindowsPath(r'C:\Users\Christoph\code\deploy\dasIT').as_posix()


####################################################################
#------------------ Preset Transducer and Medium ------------------#

# Load Verasonix Setup
ARG_DATA_PATH = os.path.join(ARG_BASE_PATH, 'data/2024_USDataRecycler_debuging')
ARG_TRANSDUCER_PATH = os.path.join(ARG_BASE_PATH, r'data\GE9LD_transducer\GE9LD.mat')
ARG_PINMAP_PATH = os.path.join(ARG_BASE_PATH, r'data\GE9LD_transducer\GE9LD_pinmap.csv')



# dasIT transducer
physical_transducer = TDloader('example_data/CIRSphantom_GE9LD_VVantage/transducer.csv')
dasIT_transducer = transducer(center_frequency_hz = 5.3e6,  # <--- FILL IN CENTER FREQUENCY OF THE TRANSDUCER IN [Hz]
                              bandwidth_hz=physical_transducer.transducer['bandwidth'].dropna().to_numpy(dtype='float', copy=False),    # [Hz]
                              adc_ratio=4,  # [-]
                              transducer_elements_nr = 192, # <--- FILL IN THE NUMBER OF TRANSDUCER ELEMENTS [#]
                              element_pitch_m = 2.3e-4, # <--- FILL IN THE ELEMENT PITCH IN [m]
                              pinmap=physical_transducer.transducer['pinmap'].dropna().to_numpy(dtype='int', copy=False),   # [-]
                              pinmapbase=1, # [-]
                              elevation_focus=0.028, # [m]
                              focus_number=None,
                              totalnr_planewaves=1,     # [-]
                              planewave_angle_interval=[0,0],   # [rad]
                              axial_cutoff_wavelength=5,  # [#]
                              speed_of_sound_ms = 1540)  # <--- FILL IN THE SPEED OF SOUND IN [m/s]


# dasIT medium
dasIT_medium = medium(speed_of_sound_ms = 1540, # [m/s]
                      center_frequency = dasIT_transducer.center_frequency, # [Hz]
                      sampling_frequency = dasIT_transducer.sampling_frequency, # [Hz]
                      max_depth_wavelength = 177,   # [#]
                      lateral_transducer_element_spacing = dasIT_transducer.lateral_transducer_spacing, # [m]
                      axial_extrapolation_coef = 1.05,  # [-]
                      attenuation_coefficient= 0.75,   # [dB/(MHz^y cm)]
                      attenuation_power=1.5   # [-]
                      )

####################################################################
#------------------------- RFData Loading -------------------------#

### Load RF Data
ARG_RFDATA_PATH = os.path.join(ARG_BASE_PATH, r'data\usdatarecycler\original_sample_with_error\sensor_data_verasonics_2019_epfl_session_2_pw0.npy')
RFdata = np.load(ARG_RFDATA_PATH)
# RFdata = RFDataloader(ARG_RFDATA_PATH,
#                       dtype='mat',
#                       mode='single')

### Preprocess (Clip and Sort) RF Data
# SAMPLE START: samples start at first recorded echo (number of wavelength distance is provided from vendor) -> null out the rest to not oveshadow the real results
# samples end at penetration depth -> clip rest of samples without data
# sort the transducer pin map
RFdata.signal[:dasIT_transducer.start_depth_rec_samples, :, :] = 0
RFdata.signal = RFdata.signal[:dasIT_medium.rx_echo_totalnr_samples, dasIT_transducer.transducer_pinmap, :]


####################################################################
#---------------------- Time Gain Compensation --------------------#

# Load tgc-waveform
ARG_TGC_PT_PATH = os.path.join(ARG_BASE_PATH, r'data\tgc_cntrl_pt.csv')
ARG_TGC_WF_PATH = os.path.join(ARG_BASE_PATH, r'data\tgc_waveform.csv')
tgc_cntrl_points = TGCloader(controlpt_path=ARG_TGC_PT_PATH)

TGCsignals = tg_compensation(signals=RFdata.signal,
                             medium=dasIT_medium,
                             center_frequency=dasIT_transducer.center_frequency,
                             cntrl_points=tgc_cntrl_points,
                             mode='points')


####################################################################
#---------------------------- Filtering ---------------------------#

### Filter RF Data
RFdata_filtered = RFfilter(signals=TGCsignals.signals,
                           fcutoff_band=dasIT_transducer.bandwidth,
                           fsampling=dasIT_transducer.sampling_frequency,
                           type='gaussian',
                           order=10)

# Plot Filter Results of all channels
# transducer_channel_map(RFdata.signal[:,0,0], RFdata.signal.shape[1], cmode='mono')
# Plot Filter Results of single channel
# amp_freq_1channel(RFdata.signal[:,0,0],
#                   fftsignal(RFdata.signal[:,0,0], ARG_ADC_SAMPLING_FQ),
#                   RFdata_filtered.signal[:,0,0],
#                   fftsignal(RFdata_filtered.signal[:,0,0], ARG_ADC_SAMPLING_FQ),
#                   mode='overlay')

# plot_signal_image(RFdata_filtered.signal[:,:,0], compression=True, dbrange=60)



# filtered_channels_matlab_file = pathlib.PureWindowsPath(r'C:\Users\Christoph\OneDrive\003_USLOCOMOTOR_GitREPO\dasIT\data\filteredchannel.csv').as_posix()
# filtered_channels_matlab = np.expand_dims(pd.read_csv(filtered_channels_matlab_file, header=None).to_numpy(),axis=2)

####################################################################
#------------------------ Analytical Signal -----------------------#

### Hilbert Transform
# RFdata_analytic = analytic_signal(np.squeeze(filtered_channels_matlab), interp=False)
RFdata_analytic = analytic_signal(np.squeeze(RFdata_filtered.signal), interp=False)

# IQsignal_1ch(RFdata_analytic,
#              RFdata_filtered,
#              mode='window',
#              start=400,
#              stop=600)

plot_signal_image(RFdata_analytic[:,:,0], compression=True, dbrange=35, path=ARG_DATA_PATH)


####################################################################
#-------------------------- Apodization Table --------------------------#

apodization = apodization(delays=None,
                          medium=dasIT_medium.medium,
                          transducer=dasIT_transducer,
                          apo='rec',
                          angles=dasIT_transducer.planewave_angles())


####################################################################
#-------------------------- Delay Tables --------------------------#

### DAS delay tabels for tilted planewaves
delay_table = planewave_delays(medium=dasIT_medium.medium,
                               sos=dasIT_medium.speed_of_sound,
                               fsampling=dasIT_transducer.sampling_frequency,
                               angles=dasIT_transducer.planewave_angles())


#plot_signal_grid(delay_table.sample_delays[:,:,0], dasIT_medium.medium, compression=False, dbrange=60)


####################################################################
#-------------------------- Beamforming ---------------------------#
start_das_timing = datetime.now()

# Mask images areas in axial direction which have been included for reconstruction
# but are not part of the actual image.
RFsignals = RFdata_analytic[:,:,0]

RFsignals = np.expand_dims(RFsignals, 2)
RFsignals = np.repeat(RFsignals, RFsignals.shape[1], axis=2)
RFsignals = np.expand_dims(RFsignals, 3)

BFsignals = RXbeamformer(signals=RFsignals,
                         delays=delay_table.sample_delays,
                         apodization=apodization.table)

# BFsignals = RXbeamformer(signals=RFsignals,
#                          delays=delay_table.sample_delays)


####################################################################
#------------------------ Image Formation --------------------------

# Envelope
BFsignals.envelope = abs(BFsignals.frame)

# Interpolate over Lateral space
BFsignals.interpolated = interp_lateral(signals=BFsignals.envelope,
                                        transducer=dasIT_transducer,
                                        medium=dasIT_medium,
                                        scale=3)


# Plot image
plot_signal_grid(signals=BFsignals.interpolated.signals_lateral_interp,
                 axis_vectors_xz=BFsignals.interpolated.imagegrid_mm,
                 axial_clip=[dasIT_transducer.start_depth_rec_m, None],
                 compression=True,
                 dbrange=58,
                 path=ARG_DATA_PATH)

IQsignal_1ch(BFsignals.frame[:,128].real, mode='full', path=ARG_DATA_PATH)

