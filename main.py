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
from dasIT.features.image import interpolate_bf
from dasIT.visualization.signal_callback import amp_freq_1channel, transducer_channel_map, IQsignal_1ch
from dasIT.visualization.image_callback import plot_signal_grid, plot_signal_image

####################################################################
#-------------------- Preset MANUAL / USERINPUT -------------------#
ARG_BASE_PATH = pathlib.PureWindowsPath(r'C:\Users\christoph\code\deploy\dasIT').as_posix()


####################################################################
#------------------ Preset Transducer and Medium ------------------#

# Load Verasonix Setup
ARG_RES_PATH = os.path.join(ARG_BASE_PATH, 'data/results/2024_USDataRecycler_debuging')
ARG_TRANSDUCER_PATH = os.path.join(ARG_BASE_PATH, r'example_data\CIRSphantom_GE9LD_VVantage\transducer.csv')
ARG_TGC_PT_PATH = os.path.join(ARG_BASE_PATH, r'example_data\CIRSphantom_GE9LD_VVantage\tgc_cntrl_pt.csv')
ARG_TGC_WF_PATH = os.path.join(ARG_BASE_PATH, r'example_data\CIRSphantom_GE9LD_VVantage\tgc_waveform.csv')



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
                              focus_number=0.5,
                              totalnr_planewaves=1,     # [-]
                              planewave_angle_interval=[0,0],   # [rad]
                              axial_cutoff_wavelength=5,  # [#]
                              speed_of_sound_ms = 1540)  # <--- FILL IN THE SPEED OF SOUND IN [m/s]

# dasIT medium
dasIT_medium = medium(speed_of_sound_ms = 1540, # [m/s]
                      center_frequency = dasIT_transducer.center_frequency, # [Hz]
                      sampling_frequency = dasIT_transducer.sampling_frequency, # [Hz]
                      max_depth_wavelength = 176,   # [#] 177
                      lateral_transducer_element_spacing = dasIT_transducer.lateral_transducer_spacing, # [m]
                      axial_extrapolation_coef = 1.05,  # [-]
                      grid_granularity=2,
                      attenuation_coefficient= 0.75,   # [dB/(MHz^y cm)]
                      attenuation_power=1.5   # [-]
                      )

####################################################################
#------------------------- RFData Loading -------------------------#

### Load RF Data
ARG_DATA_FLAG = 3   #   FLAG==1 --> CIRS Phantom, FLAG==2 --> USDataRecycler Data

if ARG_DATA_FLAG == 1:
    class NestedArray:
      def __init__(self, signal):
        self.signal = signal
    ARG_RFDATA_PATH = os.path.join(ARG_BASE_PATH, r'data\usdatarecycler\original_sample_with_error\sensor_data_verasonics_2019_epfl_session_2_pw0.npy')
    RFdata = NestedArray(np.load(ARG_RFDATA_PATH))
    RFdata.signal = np.expand_dims(RFdata.signal, axis=2)
    # RFdata.signal=np.concatenate((RFdata.signal, np.repeat(np.zeros((1, 192, 1)), 8, axis=0)), axis=0)
    RFdata.signal[-25:,:,:] = 0
    plot_signal_image(RFdata.signal[:,:,0], compression=True, dbrange=35, path=os.path.join(ARG_RES_PATH,'rf_recycler.png'))

if ARG_DATA_FLAG == 2:
    class NestedArray:
      def __init__(self, signal):
        self.signal = signal
    ARG_RFDATA_PATH = os.path.join(ARG_BASE_PATH, r'data\usdatarecycler\fixed_sample\clipped_sensor_data.npy')
    RFdata = NestedArray(np.load(ARG_RFDATA_PATH))
    RFdata.signal = np.expand_dims(RFdata.signal, axis=2)
    # RFdata.signal=np.concatenate((RFdata.signal, np.repeat(np.zeros((1, 192, 1)), 8, axis=0)), axis=0)
    plot_signal_image(RFdata.signal[:,:,0], compression=True, dbrange=35, path=os.path.join(ARG_RES_PATH,'rf_recycler.png'))

    # a = np.copy(RFdata.signal)
    # b = np.copy(np.squeeze(a[24:(24 + 23), :, 0]))
    # b = np.expand_dims(b, axis=2)
    # a[:23, :, :] = b
    # RFdata.signal[:23] = b
    #
    # d = np.copy(np.squeeze(a[(1371 - 36):1371, :, 0]))
    # d = np.expand_dims(d, axis=2)
    # RFdata.signal[1372:] = d
    # e=np.squeeze(RFdata.signal)


elif ARG_DATA_FLAG == 3:
    ARG_RFDATA_PATH = os.path.join(ARG_BASE_PATH, r'example_data\CIRSphantom_GE9LD_VVantage\CIRS_phantom.h5')
    RFdata = RFDataloader(ARG_RFDATA_PATH)
    Frame = 0
    e = np.squeeze(RFdata.signal[:, :, Frame])
    plot_signal_image(RFdata.signal[:,:,1], compression=True, dbrange=35, path=os.path.join(ARG_RES_PATH,'rf_dasIT.png'))



### Preprocess (Clip and Sort) RF Data
# SAMPLE START: samples start at first recorded echo (number of wavelength distance is provided from vendor) -> null out the rest to not oveshadow the real results
# samples end at penetration depth -> clip rest of samples without data
# sort the transducer pin map
RFdata.signal[:dasIT_transducer.start_depth_rec_samples, :, :] = 0
RFdata.signal = RFdata.signal[:dasIT_medium.rx_echo_totalnr_samples, dasIT_transducer.transducer_pinmap, :]

####################################################################
#---------------------- Time Gain Compensation --------------------#

# Load tgc-waveform

tgc_cntrl_points = TGCloader(controlpt_path=ARG_TGC_PT_PATH)
TGCsignals = tg_compensation(signals=RFdata.signal,
                             medium=dasIT_medium,
                             center_frequency=dasIT_transducer.center_frequency,
                             cntrl_points=tgc_cntrl_points.tgc_control_points,
                             mode='points')


####################################################################
#---------------------------- Filtering ---------------------------#

### Filter RF Data
RFdata_filtered = RFfilter(signals=TGCsignals.signals,
                           fcutoff_band=dasIT_transducer.bandwidth,
                           fsampling=dasIT_transducer.sampling_frequency,
                           type='gaussian',
                           order=10)


####################################################################
#------------------------ Analytical Signal -----------------------#

### Hilbert Transform
RFdata_analytic = analytic_signal(np.squeeze(RFdata_filtered.signal), interp=False)

# if ARG_DATA_FLAG == 1:
#     plot_signal_image(RFdata_analytic[:,:], compression=True, dbrange=35, path=os.path.join(ARG_RES_PATH,'rf.png'))
# elif ARG_DATA_FLAG == 2:
#     plot_signal_image(RFdata_analytic[:,:,Frame], compression=True, dbrange=35, path=ARG_RES_PATH)


####################################################################
#-------------------------- Apodization Table --------------------------#

apodization = apodization(delays=None,
                          medium=dasIT_medium.medium,
                          transducer=dasIT_transducer,
                          apo='henning',
                          angles=dasIT_transducer.planewave_angles())

a = np.ones_like(apodization.table)


####################################################################
#-------------------------- Delay Tables --------------------------#

### DAS delay tabels for tilted planewaves
delay_table = planewave_delays(medium=dasIT_medium.medium,
                               sos=dasIT_medium.speed_of_sound,
                               fsampling=dasIT_transducer.sampling_frequency,
                               angles=dasIT_transducer.planewave_angles(),
                               max_depth_samples=dasIT_medium.rx_echo_totalnr_samples)


# plot_signal_grid(delay_table.sample_delays[:,:,0], dasIT_medium.medium, compression=False, dbrange=60)


####################################################################
#-------------------------- Beamforming ---------------------------#
start_das_timing = datetime.now()

# Mask images areas in axial direction which have been included for reconstruction
# but are not part of the actual image.
# RFsignals = RFdata_analytic[:,:,0]
RFsignals = RFdata_analytic[:,:]

RFsignals = np.expand_dims(RFsignals, 2)
RFsignals = np.repeat(RFsignals, RFsignals.shape[1], axis=2)
RFsignals = np.expand_dims(RFsignals, 3)

BFsignals = RXbeamformer(signals=RFsignals,
                         delays=delay_table.sample_delays,
                         apodization=apodization.table)

# plot_signal_image(np.abs(BFsignals.frame), compression=True, dbrange=35, path=os.path.join(ARG_RES_PATH,'image_pre_interpolate.png'))


####################################################################
#------------------------ Image Formation --------------------------

# Envelope
BFsignals.envelope = abs(BFsignals.frame)

if ARG_DATA_FLAG == 1 or ARG_DATA_FLAG == 2:
    # Interpolate over Lateral space
    BFsignals.interpolated = interpolate_bf(signals=BFsignals.envelope,
                                            transducer=dasIT_transducer,
                                            medium=dasIT_medium,
                                            axial_scale=1,
                                            lateral_scale=2)


    # Plot image
    plot_signal_grid(signals=BFsignals.interpolated.signals_interp,
                     axis_vectors_xz=BFsignals.interpolated.imagegrid_mm,
                     axial_clip=[dasIT_transducer.start_depth_rec_m, None],
                     compression=True,
                     dbrange=58,
                     path=os.path.join(ARG_RES_PATH,'image.png'))


    # IQsignal_1ch(BFsignals.frame[:,128].real, mode='full', path=ARG_RES_PATH)

elif ARG_DATA_FLAG == 3:
    # Interpolate over Lateral space
    BFsignals.interpolated = interpolate_bf(signals=BFsignals.envelope[:,:,Frame],
                                            transducer=dasIT_transducer,
                                            medium=dasIT_medium,
                                            axial_scale=1,
                                            lateral_scale=2)

    # Plot image
    plot_signal_grid(signals=BFsignals.interpolated.signals_interp,
                     axis_vectors_xz=BFsignals.interpolated.imagegrid_mm,
                     axial_clip=[dasIT_transducer.start_depth_rec_m, None],
                     compression=True,
                     dbrange=58,
                     path=os.path.join(ARG_RES_PATH, 'image.png'))

    # IQsignal_1ch(BFsignals.frame[:,128].real, mode='full', path=ARG_RES_PATH)


