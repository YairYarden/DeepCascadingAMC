import numpy as np
from typing import Tuple


class SNREncoder(object):
    """ Encode the SNRS as an index of the "one-hot" used in "Data-Segmenting" and "Label-Smoothing".

    Args:
        min_snr (int): Start of SNR interval. The interval includes this value. The default start value is -20.
        max_snr (int): End of SNR interval. The interval does not include this value,
         except in some cases where step is not an integer and floating point round-off affects the length of out.
        step_snr (int): Spacing between SNR values.
        n_snr_range (int): Number of SNR ranges.
        range_overlap (int): Length of SNR overlap between ranges.

    """

    def __init__(self, min_snr=-20, max_snr=18, step_snr=2, n_snr_ranges=3, snr_range_overlap=4,
                 label_name: str = 'SNR'):
        self._min_snr = min_snr
        self._max_snr = max_snr
        self._step_snr = step_snr
        self._n_snr_ranges = n_snr_ranges
        self._snr_range_overlap = snr_range_overlap
        self._label_name = label_name

        self._n_snr = 1 + (max_snr - min_snr) // step_snr

        self._snr_range_len = self.create_snr_range_lengths()
        self._ds_encoding = self.create_data_segments_encoding()
        self._ls_encoding = self.create_label_smoothing_encoding()

        self._ds_encoder, self._ls_decoder = self.create_encoder_decoder(encodings_type='ds')
        self._ls_encoder, self._ls_decoder = self.create_encoder_decoder(encodings_type='ls')

    @property
    def label_name(self) -> str:
        """The name of the column in the dataset that is categorically encoded by this
        class.
        """
        return self._label_name

    def create_snr_range_lengths(self):
        snr_range_len = np.zeros(self._n_snr_ranges, dtype=int)
        for i in range(self._n_snr_ranges):
            snr_range_len[i] = (self._n_snr - np.sum(snr_range_len)) // (self._n_snr_ranges - i)

        return snr_range_len

    @property
    def labels(self) -> Tuple[str]:
        plot_labels = list()
        min_snr = self._min_snr
        for i in range(self._n_snr_ranges):
            max_snr = min_snr + (self._snr_range_len[i] - 1) * self._step_snr
            label = f'[{min_snr},{max_snr}]'
            plot_labels.append(label)
            print(label)
            min_snr += self._snr_range_len[i] * self._step_snr
        return tuple(plot_labels)

    def create_data_segments_encoding(self):
        ds_encoding = np.zeros((self._n_snr, self._n_snr_ranges), dtype=int)

        acc_ind = 0
        for i in range(self._n_snr_ranges):
            ds_encoding[acc_ind: acc_ind + self._snr_range_len[i], i] = 1
            acc_ind += self._snr_range_len[i]

        return ds_encoding

    def create_label_smoothing_encoding(self):
        ls_encoding = self._ds_encoding.copy()

        acc_ind = 0
        for i in range(self._n_snr_ranges - 1):
            ls_encoding[acc_ind + self._snr_range_len[i] - self._snr_range_overlap // 2:
                        acc_ind + self._snr_range_len[i], i + 1] = 1
            ls_encoding[acc_ind + self._snr_range_len[i]:
                        acc_ind + self._snr_range_len[i] + self._snr_range_len[i] // 2, i] = 1
            acc_ind += self._snr_range_len[i]

        return ls_encoding

    def encodings_to_labels(self, encodings):
        unique_encodings, inds = np.unique(encodings, axis=0, return_inverse=True)
        _, labels = np.unique(np.sum(encodings * np.arange(self._n_snr_ranges), axis=1) / np.sum(encodings, axis=1),
                              return_inverse=True)
        unique_stacked = np.unique(np.column_stack((labels, inds)), axis=0)
        unique_inds = unique_stacked[:, 1]
        unique_labels = unique_stacked[:, 0]
        return unique_encodings, unique_inds, unique_labels

    def create_encoder_decoder(self, encodings_type='ds'):
        encoder = dict()
        decoder = dict()

        if encodings_type == 'ds':
            encodings = self._ds_encoding
        else:
            encodings = self._ls_encoding

        unique_encodings, unique_inds, unique_labels = self.encodings_to_labels(encodings)

        for i in range(len(unique_inds)):
            decoded = unique_encodings[unique_inds[i]]
            encoder[tuple(decoded)] = unique_labels[i]
            decoder[unique_labels[i]] = decoded

        return encoder, decoder

    @property
    def n_classes(self) -> int:
        """The Number of ranges.
        """
        return self._n_snr_ranges

    def encode(self, snrs, encodings_type='ds'):
        snrs = np.asarray(snrs)
        ret = list()

        if encodings_type == 'ds':
            encodings = self._ds_encoding
            encoder = self._ds_encoder
        else:
            encodings = self._ls_encoding
            encoder = self._ls_encoder

        snrs_id = (snrs - self._min_snr) // self._step_snr
        snrs_encodings = encodings[snrs_id, :]

        for decoded in snrs_encodings:
            ret.append(encoder[tuple(decoded)])

        return ret

    def decode(self, encodings, encodings_type='ds'):
        ret = list()

        if encodings_type == 'ds':
            decoder = self._ds_decoder
        else:
            decoder = self._ls_decoder

        for e in encodings:
            ret.append(decoder[e])
        return ret

    def check_ls_labels_in_range(self, ls_labels, range_num):
        decodeds = np.array(self.decode(ls_labels, encodings_type='ls'))
        return decodeds[:, range_num].astype(int) == 1
