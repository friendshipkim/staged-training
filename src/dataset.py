import torch
import os
import glob
import random
import logging
import multiprocessing

# import tensorflow as tf
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
global_args = None


# the dataset object we are using
class MMapTextDataset(Dataset):
    def __init__(self, mmap_filename, chunk_size, bos_token_id, eos_token_id):
        # `chunk_size - 2` to reserve space for <s> and </s>
        self.num_instances = np.memmap(mmap_filename, mode="r", dtype=np.uint16).shape[0] // (chunk_size - 2)
        # defer loading the token_ids memmap until after the first __getitem__ call.
        # when spawning new processes for ddp, there is a hard limit in python < 3.8 that
        # pickle files need to be < 4GB. By waiting until after the first __getitem__ we
        # don't have to pickle the memmap
        self.token_ids = None
        self._mmap_filename = mmap_filename
        self._chunk_size = chunk_size
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id

    def __len__(self):
        return self.num_instances

    def __getitem__(self, i):
        if self.token_ids is None:
            self.token_ids = np.memmap(self._mmap_filename, mode="r", dtype=np.uint16)
        from_index = i * (self._chunk_size - 2)
        to_index = (i + 1) * (self._chunk_size - 2)
        data = np.concatenate(([self._bos_token_id], self.token_ids[from_index:to_index], [self._eos_token_id]))
        return torch.tensor(data, dtype=torch.long)

    # ========================= preprocessing code ========================= #
    @staticmethod
    def _process_file(full_fname):
        "Step 1: tokenize an input text file then save token ids into `np.memmap` shards of size `args.shard_size`"
        fname = full_fname.split("/")[-1]
        # print(f"fname: {fname}")
        if args.data_type == "tfrecord":
            log_filename = f"{args.output_dir}/logs-{fname}.log"
        elif args.data_type == "raw_text":
            log_filename = f"{args.output_dir}/logs-{args.shard_size}/{fname}.log"
        if os.path.isfile(log_filename):
            logging.info(f"Skipping {full_fname} ...")
            return  # log file already exists. Skip current file.

        if args.num_workers > 1:
            current = multiprocessing.current_process()
            process_identity = int(current._identity[0])
        else:
            process_identity = 1

        if process_identity == 1:
            logging.info(f"Processing {full_fname} ...")

        def _write_shard():
            if len(token_list) == 0:
                return
            # if token_list[-1] != MMapTextDataset.tokenizer.sep_token_id:  # handle a rare case
            #     token_list.append(MMapTextDataset.tokenizer.sep_token_id)
            if args.data_type in ["tfrecord", "s2"]:
                shared_filename = f"{args.output_dir}/{fname}.bin"
            elif args.data_type == "raw_text":
                shared_filename = f"{args.output_dir}/shards-{args.shard_size}/{fname}-{shard_count}.bin"
            else:
                raise NotImplementedError
            logging.info(f"Writing {len(token_list)} tokens to shared {shared_filename}")
            # breakpoint()
            fp = np.memmap(shared_filename, dtype=np.uint16, mode="w+", shape=len(token_list))
            fp[:] = token_list[:]
            del fp  # flush and close file

        token_list = []
        shard_count = 0
        tokens_count = 0

        if args.data_type == "raw_text":  # the input file is one doc per line
            with open(full_fname, "r") as fin:
                for line_num, line in enumerate(tqdm(fin)):
                    line = line.strip()
                    if line == "":  # drop empty lines
                        continue
                    tokens = MMapTextDataset.tokenizer.encode(line, add_special_tokens=False)  # `__getitem__` adds special tokens
                    # print(f"line number: {line_num}, {len(tokens)} tokens")

                    token_list.extend(tokens)
                    # print(f"current token list: {len(token_list)}")

                    if len(token_list) > args.shard_size:
                        # print("more tokens than shard_size")
                        _write_shard()
                        tokens_count += len(token_list)
                        token_list = []
                        shard_count += 1
                    else:
                        # print("less tokens than shard_size, add sep_token")
                        # TODO: check sep_token for gpt
                        token_list.append(MMapTextDataset.tokenizer.bos_token_id)  # sep_token_id)
                _write_shard()
                tokens_count += len(token_list)

        # elif args.data_type == "tfrecord":  # the input file is tfrecord format of the c4 dataset
        #     fin = tf.data.TFRecordDataset(full_fname)
        #     for raw_example in tqdm(iter(fin), disable=process_identity != 1):
        #         parsed = tf.train.Example.FromString(raw_example.numpy())
        #         feature_keys = set(parsed.features.feature.keys())
        #         if "text" in feature_keys:
        #             line = parsed.features.feature["text"].bytes_list.value[0].decode()  # raw text
        #             tokens = MMapTextDataset.tokenizer.encode(line, add_special_tokens=False)  # `__getitem__` adds special tokens
        #             if args.add_sep_after_doc:
        #                 tokens.append(MMapTextDataset.tokenizer.sep_token_id)
        #             token_list.extend(tokens)
        #             tokens_count += len(token_list)
        #         shard_count += 1
        #     _write_shard()

        with open(log_filename, "w") as f:
            f.write(f"Generated {tokens_count} tokens in {shard_count + 1} shards")

    @staticmethod
    def _combine_shards(output_fname, shards_list):
        "Step 2: combining memmap shards into one `train.bin` or `val.bin` file"
        total_size = 0
        for filename in shards_list:
            total_size += np.memmap(filename, mode="r", dtype=np.uint16).shape[0]
        logging.info(f"Writing {total_size} tokens to {output_fname}")
        all_token_ids = np.empty(total_size, dtype=np.uint16)
        last_token_index = 0
        for filename in tqdm(shards_list):
            shared = np.memmap(filename, mode="r", dtype=np.uint16)
            all_token_ids[last_token_index : last_token_index + len(shared)] = shared[:]
            last_token_index += len(shared)
        fp = np.memmap(output_fname, dtype=np.uint16, mode="w+", shape=total_size)
        fp[:] = all_token_ids[:]
        del fp

    @staticmethod
    def raw_text_to_mmap(local_args):
        """This is the main preprocessing function. It processes all the text files in `args.input_dir` and
        outputs two np.memmap files, one for training and one for validation with ratio `args.train_dev_split`.
        Processing each input file involves tokenizing it, sharding it into shards of size `args.shard_size`,
        then writing each shard as an np.memmap file, shuffle the shards, split them into train and dev shards,
        then combine the shards of each set into one big file (train.bin and val.bin).
        Notice that only the shards are shuffled not the instances inside each shard. Therefor, it is important
        to use `args.shard_size` that's small enough to have a good train/dev split, but also not small enough
        to end up with a huge number of shards that might be difficult to work with.
        The stream of tokens in the memmap files represents documents separated with `tokenizer.sep_token`.
        In `__getitem__`, the `tokenizer.bos_token` and `tokenizer.eos_token`
        are added. The reason for not adding them at preprocessing time is to allow different sequence lengths
        later on. Notice that this is the "FULL-SENTENCES" setting in the RoBERTa paper, Table2.
        Example running the preprocessing:
            >>> python scripts/pretrain.py --input_dir dirWithTextFiles --train_dev_split 0.05  \
                                           --shard_size  268435456  --num_preprocessing_workers 16
        """
        global args
        args = local_args
        
        if args.use_local_cache:
            local_cache_path = f"/n/home05/wk247/workspace/staged-training/local_cache/{args.tokenizer}"
            assert os.path.exists(local_cache_path), f"{local_cache_path} doesn't exist"
            MMapTextDataset.tokenizer = AutoTokenizer.from_pretrained(local_cache_path, use_fast=True)
        else:
            MMapTextDataset.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        assert len(MMapTextDataset.tokenizer) < 65535  # will use uint16 to store token ids
        all_files = glob.glob(f"{args.input_dir}/en-processed/*")
        logger.info(f"Total number of files: {len(all_files)}")
        logger.info(f"Tokenizer - {MMapTextDataset.tokenizer}")
        if os.path.exists(f"{args.output_dir}/cache/train.bin") and os.path.exists(f"{args.input_dir}/cache/val.bin"):
            logger.info("Cache already exists. Remove the cache directory to regenerate")
            return

        # make dirs
        os.makedirs(f"{args.output_dir}/cache/", exist_ok=True)
        os.makedirs(f"{args.output_dir}/shards-{args.shard_size}/", exist_ok=True)
        os.makedirs(f"{args.output_dir}/logs-{args.shard_size}/", exist_ok=True)  # log progrss to be able to resume

        # STEP1: tokenizing and saving to shards
        # run only if shards-{args.shard_size} directory is empty
        all_shards = glob.glob(f"{args.output_dir}/shards-{args.shard_size}/*.bin")
        if len(all_shards) == 0:
            if args.num_preprocessing_workers > 1:
                from multiprocessing.pool import Pool

                with Pool(args.num_preprocessing_workers) as p:
                    list(tqdm(p.imap(MMapTextDataset._process_file, all_files), total=len(all_files)))
            else:
                [MMapTextDataset._process_file(f) for f in tqdm(all_files)]
                
            if args.shard_only:
                exit()
        else:
            pass

        if args.data_type == "raw_text":  # c4 tfrecords are already sharded
            # STEP2: shuffling shards and combining them into train.bin and val.bin files
            
            # TODO: check this
            # use original c4 splits
            train_shards = glob.glob(f"{args.output_dir}/shards-{args.shard_size}/c4-train.*.bin")
            val_shards = glob.glob(f"{args.output_dir}/shards-{args.shard_size}/c4-validation.*.bin")
            
            # # use custom split
            # all_shards = glob.glob(f"{args.output_dir}/shards-{args.shard_size}/*.bin")
            # random.shuffle(all_shards)  # shuffling based on shards not individual lines
            # val_shards_count = int(args.train_dev_split * len(all_shards))
            # val_shards = all_shards[:val_shards_count]
            # train_shards = all_shards[val_shards_count:]
            
            logger.info(f"Total train shards: {len(train_shards)}")
            logger.info(f"Total validation shards: {len(val_shards)}")
            
            # TODO: if MMapTextDataset._combining_shards is very slow for large files, it can be skipped but we nned to
            # update the dataset to read from multiple shards directly
            # MMapTextDataset._combine_shards(f"{args.output_dir}/cache/val.bin", val_shards)
            MMapTextDataset._combine_shards(f"{args.output_dir}/cache/train.bin", train_shards)

        elif args.data_type == "tfrecord":
            train_shards = glob.glob(f"{args.output_dir}/*train*.bin")
            val_shards = glob.glob(f"{args.output_dir}/*val*.bin")
            MMapTextDataset._combine_shards(f"{args.output_dir}/val.bin", val_shards)
            MMapTextDataset._combine_shards(f"{args.output_dir}/train.bin", train_shards)

        del MMapTextDataset.tokenizer

    # ========================= end preprocessing code ========================= #
