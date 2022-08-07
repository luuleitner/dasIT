'''
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements. See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

Author: Christoph Leitner, Date: Aug. 2022
'''


import numpy as np
from matplotlib import pyplot as plt

def amp_1channel(signal=None, ratio=[5,3]):
    fig = plt.figure(figsize=(ratio[0], ratio[1]), dpi=300)
    ax_1 = fig.add_subplot(111)
    ax_1.plot(signal)
    ax_1.set_xlabel('Samples [#]')
    ax_1.set_ylabel('Signal [V]')
    ax_1.set_title('Channel Amplitude')

    plt.tight_layout()
    plt.show()



def amp_freq_1channel(ampRaw, fftRaw, ampFil, fftFil, mode='box'):

    if mode == 'box':
        fig = plt.figure(figsize=(10, 5), dpi=300)
        ax_1 = fig.add_subplot(221)
        ax_1.plot(ampRaw)
        ax_1.set_xlabel('Samples [#]')
        ax_1.set_ylabel('Signal [V]')
        ax_1.set_title('Channel Amplitude')

        ax_2 = fig.add_subplot(222)
        ax_2.plot(fftRaw[0], fftRaw[1])
        ax_2.set_xlabel('Frequency [Hz]')
        ax_2.set_ylabel('Power [W/Hz]')
        ax_2.set_title('Channel Frequency')

        ax_3 = fig.add_subplot(223)
        ax_3.plot(ampFil, 'r')
        ax_3.set_xlabel('Samples [#]')
        ax_3.set_ylabel('Signal [V]')

        ax_4 = fig.add_subplot(224)
        ax_4.plot(fftFil[0], fftFil[1], 'r')
        ax_4.set_xlabel('Frequency [MHz]')
        ax_4.set_ylabel('Power [W/Hz]')

        plt.tight_layout()
        plt.show()

    elif mode == 'overlay':
        fig = plt.figure(figsize=(10, 8), dpi=300)
        ax_1 = fig.add_subplot(211)
        ax_1.plot(ampRaw)
        ax_1.plot(ampFil, 'r')
        ax_1.set_xlabel('Samples [#]')
        ax_1.set_ylabel('Amplitude')
        ax_1.set_title('Channel Amplitude')

        ax_2 = fig.add_subplot(212)
        ax_2.plot(fftRaw[0], fftRaw[1])
        ax_2.plot(fftFil[0], fftFil[1], 'r')
        ax_2.set_xlabel('Frequency [Hz]')
        ax_2.set_ylabel('Power [W/Hz]')
        ax_2.set_title('Channel Frequency')

        plt.tight_layout()
        plt.show()

    else:
        print('ERROR: Wrong value encountered')


def transducer_channel_map(data, nr_channels, cmode='color'):
    n_columns = 1
    n_rows = nr_channels + 1

    if cmode == 'color':
        color = plt.cm.hsv(np.linspace(0, 1, nr_channels))
        np.random.shuffle(color)
    elif cmode == 'mono':
        color = np.full(nr_channels, 'b')
    else:
        print('ERROR: Wrong value encountered')

    fig, axs = plt.subplots(n_rows, n_columns, figsize=(8, 35), dpi=400)
    for idx, (ch_signal, c) in enumerate(zip(data.T, color)):
        axs.ravel()[idx].plot(ch_signal, linewidth=0.5, c=c)
        plt.axis('off')
        axs.ravel()[idx].set_xticks([])
        axs.ravel()[idx].set_yticks([])
        axs.ravel()[idx].axis('off')
    plt.tight_layout()
    plt.show()


def IQsignal_1ch(analytic_signal, RFdata_filtered, mode='full', start=None, stop=None):
    if mode == 'full':
        analytic_signal = analytic_signal[:]
    elif mode == 'window':
        analytic_signal = analytic_signal[start:stop]
    else:
        print('ERROR: Wrong value encountered')

    fig = plt.figure(figsize=(10, 5), dpi=300)
    ax_1 = fig.add_subplot(111)
    ax_1.plot(analytic_signal.real, 'r', linewidth=.5, alpha=0.5)
    ax_1.plot(analytic_signal.imag, 'b', linewidth=.5, alpha=0.5)
    ax_1.plot(abs(analytic_signal), 'g', linewidth=1, alpha=1)

    ax_1.set_xlabel('Samples [#]')
    ax_1.set_ylabel('Signal')
    plt.show()


