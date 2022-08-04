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
import pandas as pd
import h5py

class RFDataloader():
    def __init__(self, path):
        self.signal = self._loadH5data(path).astype(np.float64)

    def _loadH5data(self, p):
        with h5py.File(p, 'r') as file:
            nbr_frames = len(file.keys())
            nbr_shots = len(file[f'frame0000'].keys())

            frames = []
            for f in range(nbr_frames):
                shots = []
                for s in range(nbr_shots):
                    shot = file[f'frame{f:04}/shot{s:04}'][:]
                    shots.append(shot)
                frames.append(shots)
            frames = np.rollaxis(np.vstack(frames), 0, 3)

        return frames

class TDloader():
    def __init__(self, transducer_path=None):
        self.transducer = pd.read_csv(transducer_path)


class TGCloader():
    def __init__(self, controlpt_path=None):
        self.tgc_control_points = pd.read_csv(controlpt_path, header=None)