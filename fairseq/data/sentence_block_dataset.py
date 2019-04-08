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

    def __init__(self, dataset, sizes, dim_offsets, pad, eos, mask):
        super().__init__()
        self.dataset = dataset
        self.pad = pad
        self.eos = eos
        self.mask = mask

        assert len(dataset) == len(sizes)

        sizes = np.array(sizes, dtype=int)
        dim_offsets = np.array(dim_offsets, dtype=int)
        total_size = sum(sizes)
        length = len(sizes)
        self.length = length

        # build index mapping block indices to the underlying dataset indices
        self.block_to_dataset_index = np.empty((length, 3), dtype=int)
        ds_idx, ds_remaining = -1, 0
        print('dataset: ', dataset)
        print('sizes: ', sizes)
        print('len sizes: ', len(sizes))
        print('dim_offsets: ', dim_offsets)
        print('len dim_offsets: ', len(dim_offsets))
        print('length: ', length)
        print('shape: ', self.block_to_dataset_index.shape)     
        print('total_size: ', total_size)  
        self.src_sizes = np.empty(length, dtype=int)
        self.tgt_sizes = np.empty(length, dtype=int)
        #sys.exit()
        start_ds_idx = 0
        block_idx = 0
        prev = 0
        for idx in range(len(dim_offsets) - 1):
            #print('idx: ', idx)
            #print('idx max: ', len(dim_offsets))
            #print('block_idx: ', block_idx)
            size = dim_offsets[idx+1] - dim_offsets[idx]
            if size == 0:
                print('TODO: fix preprocess issue, e.g., encode empty line with <newline>')
                #sys.exit()
                #continue
            #print('cur size: ', size)
            ds_idx = start_ds_idx + size - 1
            for start_offset in range(size):
                self.block_to_dataset_index[block_idx] = (
                    start_ds_idx,  # starting index in dataset
                    start_offset,  # starting offset within starting index
                    ds_idx,  # ending index in dataset
                )

                target_idx = start_ds_idx + start_offset
                self.src_sizes[block_idx] = sum(sizes[start_ds_idx:target_idx]) + 1 + sum(sizes[target_idx + 1:ds_idx + 1])
                self.tgt_sizes[block_idx] = sum(sizes[start_ds_idx:ds_idx + 1])
                
                #self.sizes[block_idx] = sum(sizes[start_ds_idx:ds_idx + 1])
                #print('sizes at idx: ', self.sizes[block_idx])
                block_idx += 1
                #print('start_ds_idx: ', start_ds_idx)
                #print('ds_idx: ', ds_idx)
            #sys.exit()
            start_ds_idx = ds_idx + 1
        print('block_idx: ', block_idx) 
        print('length: ', length)
        #print('self.sizes: ', self.sizes)
        assert dim_offsets[-1] == len(sizes) 
        assert length == block_idx
        #sys.exit() 

    def __getitem__(self, index):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
        target_idx = start_ds_idx + start_offset
        target = torch.cat([self.dataset[target_idx]])

        before = torch.cat([
            self.dataset[idx] for idx in range(start_ds_idx,  target_idx)
        ]) if start_ds_idx <= target_idx - 1 else torch.empty(0, dtype=target.dtype)
        after =  torch.cat([
            self.dataset[idx] for idx in range(target_idx + 1,  end_ds_idx + 1)
        ]) if target_idx + 1 <= end_ds_idx else torch.empty(0, dtype=target.dtype)
        item = torch.cat([
            before, target, after
        ])
        source = torch.cat([
            before, item.new([self.mask]), after
        ])      
        print('idx: ', index)
        print('target: ', target)
        print('item: ', item)
        print('source: ', source)
        #sys.exit()
        # *item* is the original sentences target + context (=item)
        # *source* is sentences without the target sentence, and with the target position delimiter
        # *target* is the target sentence, target only
        #self.sizes[block_idx][0] = source.size()
        #self.sizes[block_idx][1] = item.size()
        #print('source size: ', source.size())
        #print('target size: ', item.size())
        #print('total size: ', self.sizes[index])
        assert self.src_sizes[index] == source.size()
        assert self.tgt_sizes[index] == item.size()
        #sys.exit()
        return source, item, target

    def __len__(self):
        return self.length

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
