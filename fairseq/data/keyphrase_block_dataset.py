# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import numpy as np
import torch
import random

from . import FairseqDataset


class KeyphraseBlockDataset(FairseqDataset):
    """Break a Dataset of tokens into blocks.
    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes (List[int]): sentence lengths (required for 'complete' and 'eos')
    """
    def __init__(self, dataset, sizes, dim_offsets, pad, stopwords, eos, mask):
        super().__init__()
        self.dataset = dataset
        self.pad = pad
        self.eos = eos
        self.mask = mask
        self.stopwords = stopwords

        assert len(dataset) == len(sizes)
         
        random.seed(9)
        sizes = np.array(sizes, dtype=int)
        dim_offsets = np.array(dim_offsets, dtype=int)
        total_size = sum(sizes)
        length = len(sizes)
        self.length = length

        # build index mapping block indices to the underlying dataset indices
        self.block_to_dataset_index = []#np.empty((length, 3), dtype=int)
        self.src_sizes = []#np.empty(length, dtype=int)
        self.tgt_sizes = []#np.empty(length, dtype=int)
        #self.block_to_dataset_index = np.empty((length, 3), dtype=int)
        ds_idx, ds_remaining = -1, 0
        #print('dataset: ', dataset)
        print('sizes: ', sizes)
        print('len sizes: ', len(sizes))
        print('dim_offsets: ', dim_offsets)
        print('len dim_offsets: ', len(dim_offsets))
        print('length: ', length)
        #print('shape: ', self.block_to_dataset_index.shape)     
        print('total_size: ', total_size)
        #sys.exit() 
        #self.src_sizes = np.empty(length, dtype=int)
        #self.tgt_sizes = np.empty(length, dtype=int)
        #sys.exit()
        start_ds_idx = 0
        block_idx = 0
        prev = 0
        invalid = 0
        batch_idxes = []
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
                if random.uniform(0, 1) > 0.2:
                    continue

                self.block_to_dataset_index.append((
                    start_ds_idx,  # starting index in dataset
                    start_offset,  # starting offset within starting index
                    ds_idx,  # ending index in dataset
                ))

                target_idx = start_ds_idx + start_offset
                self.src_sizes.append(sum(sizes[start_ds_idx:target_idx]) + 3 + sum(sizes[target_idx + 1:ds_idx + 1]))
                #print('story len: ', sum(sizes[start_ds_idx:ds_idx + 1]))
                #invalid += 1 if sum(sizes[start_ds_idx:ds_idx + 1]) > 1022 else 0 
                # setting 1: incldue context in the target
                #self.tgt_sizes[block_idx] = sum(sizes[start_ds_idx:ds_idx + 1])
                # setting 2: target only
                self.tgt_sizes.append(2)#sum(sizes[target_idx:target_idx + 1]))
                 
                #self.sizes[block_idx] = sum(sizes[start_ds_idx:ds_idx + 1])
                #print('sizes at idx: ', self.sizes[block_idx])
                if start_ds_idx <= target_idx - 1 or target_idx + 1 <= ds_idx:
                    batch_idxes.append(block_idx)
                
                block_idx += 1
                #print('start_ds_idx: ', start_ds_idx)
                #print('ds_idx: ', ds_idx)
            #sys.exit()
            start_ds_idx = ds_idx + 1
        print('block_idx: ', block_idx) 
        print('length: ', length)
        print('batch_idxes: ', len(batch_idxes))
        batch_idxes = torch.tensor(batch_idxes)
        #print('self.block_to_dataset_index: ', self.block_to_dataset_index)
        
        tmp_block = np.empty((block_idx, 3), dtype=int)
        tmp_src_sizes = np.empty(block_idx, dtype=int)
        tmp_tgt_sizes = np.empty(block_idx, dtype=int)

        for idx in range(block_idx):
            tmp_block[idx] = self.block_to_dataset_index[idx]
            tmp_src_sizes[idx] = self.src_sizes[idx]
            tmp_tgt_sizes[idx] = self.tgt_sizes[idx]

        self.block_to_dataset_index = tmp_block[batch_idxes]                
        self.src_sizes = tmp_src_sizes[batch_idxes]
        self.tgt_sizes = tmp_tgt_sizes[batch_idxes]
        #self.block_to_dataset_index = self.block_to_dataset_index[batch_idxes]
        #self.src_sizes = self.src_sizes[batch_idxes]
        #self.tgt_sizes = self.tgt_sizes[batch_idxes]
        self.length = len(self.block_to_dataset_index)
        print('block_idx: ', block_idx)
        print('self.block_to_dataset_index: ', len(self.block_to_dataset_index))
        print('self.src_sizes: ', len(self.src_sizes))
        print('self.tgt_sizes: ', len(self.tgt_sizes))
        #print('self.sizes: ', self.sizes)
        assert dim_offsets[-1] == len(sizes) 
        #assert length == block_idx
        #print('invalid: ', invalid)
        #sys.exit() 

    def old(self, dataset, sizes, dim_offsets, stopwords, pad, eos, mask):
        super().__init__()
        self.dataset = dataset
        self.pad = pad
        self.eos = eos
        self.mask = mask
        self.stopwords = stopwords

        assert stopwords != None

        assert len(dataset) == len(sizes)
        random.seed(9)
        
        sizes = np.array(sizes, dtype=int)
        dim_offsets = np.array(dim_offsets, dtype=int)
        total_size = sum(sizes)
        #length = len(sizes)
        #self.length = length

        # build index mapping block indices to the underlying dataset indices
        self.block_to_dataset_index = []#np.empty((length, 3), dtype=int)
        ds_idx, ds_remaining = -1, 0
        #print('dataset: ', dataset)
        print('sizes: ', sizes)
        print('len sizes: ', len(sizes))
        print('dim_offsets: ', dim_offsets)
        print('len dim_offsets: ', len(dim_offsets))
        #print('length: ', length)
        #print('shape: ', self.block_to_dataset_index.shape)     
        print('total_size: ', total_size)
        #print('dataset: ', dataset)
        #sys.exit() 
        self.src_sizes = []#np.empty(length, dtype=int)
        self.tgt_sizes = []#np.empty(length, dtype=int)
        #sys.exit()
        start_ds_idx = 0
        block_idx = 0
        prev = 0
        invalid = 0
        batch_idxes = []
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
                if random.uniform(0, 1) > 0.2:
                    continue
                 
                #target_idx = start_ds_idx + start_offset
                #print(target_idx)
                #print(self.dataset)
                #target = torch.cat([self.dataset[target_idx]])
                #print('target: ', target)
                #nonstopword_target = []
                #for item in target:
                #    if item.item() not in self.stopwords:
                #        nonstopword_target.append(item.item())

                #print(nonstopword_target)
                #random.shuffle(nonstopword_target)
                #target_size = min(5, len(nonstopword_target))
                #nonstopword_target = nonstopword_target[:target_size]
                #target = torch.tensor(nonstopword_target)
               
                for tok_idx in range(1):
                    #print(block_idx)
                    #print(self.block_to_dataset_index)
                    self.block_to_dataset_index.append((
                        start_ds_idx,  # starting index in dataset
                        start_offset,  # starting offset within starting index
                        ds_idx,  # ending index in dataset
                    ))
                    
                    target_idx = start_ds_idx + start_offset
                    self.src_sizes.append(sum(sizes[start_ds_idx:target_idx]) + 3 + sum(sizes[target_idx + 1:ds_idx + 1]))
                    #print('story len: ', sum(sizes[start_ds_idx:ds_idx + 1]))
                    #invalid += 1 if sum(sizes[start_ds_idx:ds_idx + 1]) > 1022 else 0 
                    # setting 1: incldue context in the target
                    #self.tgt_sizes[block_idx] = sum(sizes[start_ds_idx:ds_idx + 1])
                    # setting 2: target only
                    self.tgt_sizes.append(1)#sum(sizes[target_idx:target_idx + 1])
                 
                    #self.sizes[block_idx] = sum(sizes[start_ds_idx:ds_idx + 1])
                    #print('sizes at idx: ', self.sizes[block_idx])
                    if start_ds_idx <= target_idx - 1 or target_idx + 1 <= ds_idx:
                        batch_idxes.append(block_idx)
                    block_idx += 1
                    #print('start_ds_idx: ', start_ds_idx)
                    #print('ds_idx: ', ds_idx)
            #sys.exit()
            start_ds_idx = ds_idx + 1
        print('block_idx: ', block_idx) 
        #print('length: ', length)
        print('batch_idxes: ', len(batch_idxes))
        batch_idxes = torch.tensor(batch_idxes)
        #print('self.block_to_dataset_index: ', self.block_to_dataset_index)
        tmp_block = np.empty((block_idx, 3), dtype=int)
        tmp_src_sizes = np.empty(block_idx, dtype=int)
        tmp_tgt_sizes = np.empty(block_idx, dtype=int)

        for idx in range(block_idx):
            tmp_block[idx] = self.block_to_dataset_index[idx]
            tmp_src_sizes[idx] = self.src_sizes[idx]
            tmp_tgt_sizes[idx] = self.tgt_sizes[idx]

        self.block_to_dataset_index = tmp_block[batch_idxes]                
        self.src_sizes = tmp_src_sizes[batch_idxes]
        self.tgt_sizes = tmp_tgt_sizes[batch_idxes]
        self.length = len(self.block_to_dataset_index)
        print('block_idx: ', block_idx)
        print('self.block_to_dataset_index: ', len(self.block_to_dataset_index))
        print('self.src_sizes: ', len(self.src_sizes))
        print('self.tgt_sizes: ', len(self.tgt_sizes))
        #print('self.sizes: ', self.sizes)
        #assert dim_offsets[-1] == len(sizes) 
        #assert length == block_idx
        #print('invalid: ', invalid)
        #sys.exit() 

    def __getitem__(self, index):
        print('index: ', index)
        #debug = False
        #if index == 7943:
        #    debug = True
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
        target_idx = start_ds_idx + start_offset
        target_sent = torch.cat([self.dataset[target_idx]])
        #print('target: ', target)
        #print(self.stopwords)
        
        nonstopword_target = []
        for item in target_sent:
            if item.item() not in self.stopwords:
                nonstopword_target.append(item.item())
        if len(nonstopword_target) == 0:
           return None

        random.shuffle(nonstopword_target)
        tok_idx = random.randint(0, len(nonstopword_target)-1)
        target = torch.tensor([nonstopword_target[tok_idx], target_sent[-1].item()])
        print(target)
        #print(target.size())
        #sys.exit()

        before = torch.cat([
            self.dataset[idx] for idx in range(start_ds_idx,  target_idx)
        ]) if start_ds_idx <= target_idx - 1 else torch.empty(0, dtype=target_sent.dtype)
        after =  torch.cat([
            self.dataset[idx] for idx in range(target_idx + 1,  end_ds_idx + 1)
        ]) if target_idx + 1 <= end_ds_idx else torch.empty(0, dtype=target_sent.dtype)
        item = torch.cat([target])
        source = torch.cat([
            before, item.new([self.mask, self.mask, self.mask]), after
        ])      
        #print('idx: ', index)
        #print('target: ', target)
        #print('item: ', item)
        #print('source: ', source)
        #sys.exit()
        # *item* is the original sentences target + context (=item)
        # *source* is sentences without the target sentence, and with the target position delimiter
        # *target* is the target sentence, target only
        #self.sizes[block_idx][0] = source.size()
        #self.sizes[block_idx][1] = item.size()
        #print('source size: ', source.size())
        #print('target size: ', item.size())
        #print('total size: ', self.sizes[index])
        print('item: ', item.size())
        print('tgt_sizes: ', self.tgt_sizes[index])
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
