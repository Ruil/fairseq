# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    Dictionary,
    IndexedCachedSentenceDataset,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset,
    SentenceBlockDataset,
    TruncatedDictionary,
    MonoLingualPairDataset,
)

from . import FairseqTask, register_task


@register_task('seq_language_modeling')
class SeqLanguageModelingTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--sentence', default=False, action='store_true',
                            help='')
        parser.add_argument('--output-dictionary-size', default=-1, type=int,
                            help='limit the size of output dictionary')
        parser.add_argument('--self-target', action='store_true',
                            help='include self target')
        parser.add_argument('--future-target', action='store_true',
                            help='include future target')
        parser.add_argument('--past-target', action='store_true',
                            help='include past target')
        parser.add_argument('--sample-break-mode',
                            choices=['none', 'complete', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 'If set to "eos", includes only one sentence per sample.')
        # fmt: on

    @staticmethod
    def load_pretrained_model(path, src_dict_path, tgt_dict_path, arg_overrides=None):
        model = utils.load_checkpoint_to_cpu(path)
        args = model['args']
        state_dict = model['model']
        args = utils.override_model_args(args, arg_overrides)
        src_dict = Dictionary.load(src_dict_path)
        tgt_dict = Dictionary.load(tgt_dict_path)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        task = TranslationTask(args, src_dict, tgt_dict)
        model = task.build_model(args)
        model.upgrade_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

    def __init__(self, args, dictionary, output_dictionary, targets=None):
        super().__init__(args)

        self.dictionary = dictionary
        self.output_dictionary = output_dictionary

        self.sentence = args.sentence
        print('self.sentence: ', self.sentence)

        if targets is None:
            targets = ['future']
        self.targets = targets
        print(targets)
        #sys.exit()

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        dictionary = None
        output_dictionary = None
        if args.data:
            print('args.data: ', args.data)
            #sys.exit()
            args.data = args.data[0]
            path = os.path.join(args.data, 'dict.txt')
            dictionary = Dictionary.load(path)
            print('| dictionary: {} types'.format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(dictionary, args.output_dictionary_size)

        # upgrade old checkpoints
        if hasattr(args, 'exclude_self_target'):
            args.self_target = not args.exclude_self_target

        targets = []
        if getattr(args, 'self_target', False):
            targets.append('self')
        if getattr(args, 'future_target', False):
            targets.append('future')
        if getattr(args, 'past_target', False):
            targets.append('past')
        if len(targets) == 0:
            # standard language modeling
            targets = ['future']
        #print(targets)
        #sys.exit()
        return cls(args, dictionary, output_dictionary, targets=targets)

    def build_model(self, args):
        model = super().build_model(args)

        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError('Unsupported language modeling target: {}'.format(target))

        return model

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            path = os.path.join(self.args.data, split_k)

            if self.args.raw_text and IndexedRawTextDataset.exists(path):
                ds = IndexedRawTextDataset(path, self.dictionary)
            elif not self.args.raw_text and IndexedDataset.exists(path):
                if self.args.lazy_load:
                    ds = IndexedDataset(path, fix_lua_indexing=True)
                elif self.sentence:
                    print('load IndexedCacheSentenceDataset')
                    ds = IndexedCachedSentenceDataset(path, fix_lua_indexing=True)
                else:
                    print('load IndexedCacheDataset')
                    ds = IndexedCachedDataset(path, fix_lua_indexing=True)
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

            loaded_datasets.append(
                SentenceBlockDataset(
                    ds, ds.sizes, ds.dim_offsets, pad=self.dictionary.pad(), eos=self.dictionary.eos(),
                    mask=self.dictionary.mask(),
                )

            )

            print('| {} {} {} examples'.format(self.args.data, split_k, len(loaded_datasets[-1])))

            if not combine:
                break 

        if len(loaded_datasets) == 1:
            dataset = loaded_datasets[0]
            #print('sizes 1: ', sizes)
        else:
            sys.exit()
            dataset = ConcatDataset(loaded_datasets)
            #print('sizes 2: ', sizes)

        #print(dataset.sizes)

        #sys.exit()
        add_eos_for_other_targets = self.args.sample_break_mode is not None and self.args.sample_break_mode != 'none'

        self.datasets[split] = MonoLingualPairDataset(
            dataset, dataset.src_sizes, self.dictionary,
            dataset.tgt_sizes, self.output_dictionary,
            add_eos_for_other_targets,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            targets=self.targets
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        sys.exit()
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.output_dictionary
