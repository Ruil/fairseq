# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import numpy as np
import torch

from . import FairseqDataset


class SentenceBlockDataset(FairseqDataset):
    """Break a Dataset of tokens into blocks.
    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes (List[int]): sentence lengths (required for 'complete' and 'eos')
    """

    def __init__(self, dataset, sizes, dim_offsets, pad, eos):
        super().__init__()
        self.dataset = dataset
        self.pad = pad
        self.eos = eos

        assert len(dataset) == len(sizes)

        sizes = np.array(sizes, dtype=int)
        total_size = sum(sizes)
        length = total_size

        # build index mapping block indices to the underlying dataset indices
        self.block_to_dataset_index = np.empty((length, 3), dtype=int)
        ds_idx, ds_remaining = -1, 0
        print('dim_offsets: ', dim_offsets)
        print('len dim_offsets: ', len(dim_offsets))
        start_ds_idx = 0
        i = 0
        for size in dim_offsets:
            if size == 0:
                print('fix preprocess issue')
                sys.exit()
                continue
            ds_idx = start_ds_idx + size - 1
            for start_offset in range(size):
                self.block_to_dataset_index[i] = (
                    start_ds_idx,  # starting index in dataset
                    start_offset,  # starting offset within starting index
                    ds_idx,  # ending index in dataset
                )
                i += 1
            start_ds_idx = ds_idx + 1

        assert ds_idx == length - 1

    def __getitem__(self, index):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
        buffer = torch.cat([
            self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)
        ])
        slice_s, slice_e = self.slice_indices[index]
        length = slice_e - slice_s
        s, e = start_offset, start_offset + length
        item = buffer[s:e]

        if self.include_targets:
            # *target* is the original sentence (=item)
            # *source* is rotated left by 1 (maybe left-padded with eos)
            # *past_target* is rotated left by 2 (left-padded as needed)
            if s == 0:
                source = torch.cat([item.new([self.eos]), buffer[0:e - 1]])
                past_target = torch.cat([item.new([self.pad, self.eos]), buffer[0:e - 2]])
            else:
                source = buffer[s - 1:e - 1]
                if s == 1:
                    past_target = torch.cat([item.new([self.eos]), buffer[0:e - 2]])
                else:
                    past_target = buffer[s - 2:e - 2]

            return source, item, past_target
        return item

    def __len__(self):
        return len(self.slice_indices)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        self.dataset.prefetch({
            ds_idx
            for index in indices
            for start_ds_idx, _, end_ds_idx in [self.block_to_dataset_index[index]]
            for ds_idx in range(start_ds_idx, end_ds_idx + 1)
        })
