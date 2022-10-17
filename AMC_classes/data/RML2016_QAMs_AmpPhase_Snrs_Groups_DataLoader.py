"""RML2016 Data loader for QAM modulation divided to 3 groups of SNR
"""

# External Includes
from collections import defaultdict
import numpy as np
import os
import pickle
import tarfile
from typing import Tuple, Dict
from urllib.request import urlretrieve
from warnings import warn
import math

# Internal Includes
from dataset import Dataset
from dataset_builder import DatasetBuilder


class RML2016_QAMs_AmpPhase_SNR_groups_DataLoader(object):
    def __init__(
        self, cache_path: str, remote_url: str, unpickled_path: str, warning_msg: str
    ):
        self.CACHE_PATH = cache_path
        self.REMOTE_URL = remote_url
        self.UNPICKLED_PATH = unpickled_path
        self.WARNING_MSG = warning_msg

    def load(self, path: str):
        if path is not None:
            if not os.path.exists(path):
                raise ValueError(
                    "If path is provided, it must actually exist.  Provided path: "
                    "{}".format(path)
                )
            return self._load_local(path=path)

        # If this function has previously been called before to fetch the dataset from the
        # remote, then it will have already been cached locally and unpickled.
        if os.path.exists(self.UNPICKLED_PATH):
            return self._load_local(self.UNPICKLED_PATH)

        warn(self.WARNING_MSG)
        self._download()
        return self._load_local(self.UNPICKLED_PATH)

    def _load_local(self, path: str) -> Dataset:
        builder_lowSNR = DatasetBuilder()
        builder_mediumSNR = DatasetBuilder()
        builder_highSNR = DatasetBuilder()
        data, description = self._read(path)
        for mod, snrs in description.items():
            if(mod != "QAM16" and mod != "QAM64"):
                continue
                
            for snr in snrs:
                for iq in data[(mod, snr)]:
                    maxAmp = -1
                    for iSample in range(128):
                        currI = iq[0, iSample]
                        currQ = iq[1, iSample]
                        currAmp = math.sqrt(currI * currI + currQ * currQ)
                        currPhase = np.arctan(currQ / currI)
                        iq[0, iSample] = currAmp
                        iq[1, iSample] = currPhase
                        if(abs(currAmp) > maxAmp):
                            maxAmp = abs(currAmp)
                    
                    for iSample in range(128):
                        iq[0, iSample] = iq[0, iSample] / maxAmp
                    
                
                    if(snr <= -10):
                        builder_lowSNR.add(iq=iq, Modulation=mod, SNR=snr)
                    elif(snr <= 4):
                        builder_mediumSNR.add(iq=iq, Modulation=mod, SNR=snr)
                    else:
                        builder_highSNR.add(iq=iq, Modulation=mod, SNR=snr)
                                     
        return builder_lowSNR.build(), builder_mediumSNR.build(), builder_highSNR.build() 

    def _download(self):
        urlretrieve(self.REMOTE_URL, self.CACHE_PATH)
        with tarfile.open(self.CACHE_PATH, "r:bz2") as tar:
            tar.extractall()

    def _read(self, path: str) -> Tuple[np.ndarray, Dict]:
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin")

            description = defaultdict(list)
            # Declare j just to get the linter to stop complaining about the lamba below
            j = None
            snrs, mods = map(
                lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1, 0]
            )
            for mod in mods:
                for snr in snrs:
                    description[mod].append(snr)

            return data, description


def load_RML201610B_onlyQAMs_AmpPhase_dividedSNR(path: str = None) -> Dataset:
    """Load the RadioML2016.10B Dataset provided by DeepSig Inc.

    This dataset is licensed under Creative Commons Attribution - NonCommercial -
    ShareAlike 4.0 License (CC BY-NC-SA 4.0) by DeepSig Inc.

    Args:
        path (str, optional): Path to the dataset which has already been downloaded from
                              DeepSig Inc., saved locally, and extracted (tar xjf).  If
                              not provided, the dataset will attempt to be downloaded
                              from the internet and saved locally -- subsequent calls
                              would read from that cached dataset that is fetched.

    Raises:
        ValueError: If *path* is provided but does not exist.

    Returns:
        Dataset: A Dataset that has been loaded with the data from RML2016.10B

    License
        https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

    Download Location
        https://www.deepsig.io/datasets

    Citation
        T. J. O’Shea and N. West, “Radio machine learning dataset generation with GNU
        Radio” in Proceedings of the GNU Radio Conference, vol. 1, 2016.
    """
    CACHE_PATH = "./RML2016.10b.tar.bz2"
    WARNING_MSG = """
    About to attempt downloading the RML2016.10B dataset from deepsig.io/datasets.
    Depending on your network connection, this process can be slow and error prone.  Any
    errors raised during network operations are not silenced and will therefore cause your
    code to crash.  If you require robustness in your experimentation, you should manually
    download the file locally and pass the file path to the load_RML201610b_dataset
    function.

    Further, this dataset is provided by DeepSig Inc. under Creative Commons Attribution
    - NonCommercial - ShareAlike 4.0 License (CC BY-NC-SA 4.0).  By calling this function,
    you agree to that license -- If an alternative license is needed, please contact DeepSig
    Inc. at info@deepsig.io
    """
    REMOTE_URL = "http://opendata.deepsig.io/datasets/2016.10/RML2016.10b.tar.bz2"
    UNPICKLED_PATH = "RML2016.10b.dat"

    loader = RML2016_QAMs_AmpPhase_SNR_groups_DataLoader(
        cache_path=CACHE_PATH,
        remote_url=REMOTE_URL,
        unpickled_path=UNPICKLED_PATH,
        warning_msg=WARNING_MSG,
    )
    return loader.load(path=path)
