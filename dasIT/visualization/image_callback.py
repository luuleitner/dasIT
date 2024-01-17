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


import os
import numpy as np
from matplotlib import pyplot as plt
from ..features.signal import logcompression, envelope


def plot_signal_image(signal, compression=True, dbrange=1, path=None):
    if compression:
        signal = logcompression(signal, dbrange)

    fig = plt.figure(figsize=(6, 7), dpi=300)
    ax_1 = fig.add_subplot(111)
    ax_1.imshow(signal[90:-700, :],
                aspect=1,
                cmap='gray')

    ax_1.set_xlabel('Transducer Element [#]', fontsize=15, fontweight='bold', labelpad=10)
    ax_1.set_ylabel('Passing Time (Sample [#])', fontsize=15, fontweight='bold', labelpad=10)
    ax_1.xaxis.tick_top()
    ax_1.xaxis.set_label_position('top')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    if path:
        fig.savefig(os.path.join(path, 'plot1.png'), dpi=300)


def plot_signal_grid(signals=None,
                     axis_vectors_xz=None,
                     axial_clip=None,
                     compression=True,
                     dbrange=50,
                     path=None,
                     pad=False):

    m2mm_conversion = 1000

    if axial_clip:
        # convert to mm
        axial_clip = np.array(axial_clip, dtype=np.float64)
        axial_clip[np.isnan(axial_clip)] = 0
        axial_clip *= m2mm_conversion

        # convert the percentage to clip off
        cutoff_pzt = axial_clip / np.amax(axis_vectors_xz[1])
        if cutoff_pzt[1] == 0:
            cutoff_pzt[1] = cutoff_pzt[0] * 1.5

        # calculate the number of samples to clip off
        axial_clip = np.ceil(cutoff_pzt * axis_vectors_xz[1].size).astype(np.int32)

        # clip
        axis_vectors_xz[1] = axis_vectors_xz[1][axial_clip[0]:-axial_clip[1]]
        signals = signals[axial_clip[0]:-axial_clip[1], :]


    if compression:
        signal = logcompression(signals, dbrange)
    else:
        signal = signals

    if pad:
        signal = np.pad(signal, 
                        ((axial_clip[0], axial_clip[1]), (0,0)),
                        mode='constant', 
                        constant_values=0)
        
    if path:
        # fig = plt.figure(figsize=(6, 7), dpi=300)
        fig = plt.figure(figsize=(5, 6), dpi=250)
        ax_1 = fig.add_subplot(111)
        ax_1.imshow(np.flipud(signal),
                    aspect=1,
                    interpolation='none',
                    extent=extents(axis_vectors_xz[0]) + extents(axis_vectors_xz[1]),
                    origin='upper',
                    cmap='gray')

        ax_1.invert_yaxis()

        ax_1.set_xlabel('Lateral [mm]', fontsize=15, fontweight='bold', labelpad=10)
        ax_1.set_ylabel('Axial [mm]', fontsize=15, fontweight='bold', labelpad=10)
        ax_1.xaxis.tick_top()
        ax_1.xaxis.set_label_position('top')
        ax_1.minorticks_on()

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        # plt.show()

        fig.savefig(path, dpi=300)

    return signal

def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]

