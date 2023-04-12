import argparse


def create_argument_parser():

    parser = argparse.ArgumentParser(description="pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=3)

    # custom args
    parser.add_argument(
        "--use_local_cache",
        action="store_true",
        default=False,
        help="if https connection is blocked, load tokenizer and model locally",
    )

    # Dataset. Some of these params are only useful when generating the dataset cache
    parser.add_argument("--input_dir", type=str, default="/n/home05/wk247/workspace/staged-training/local_cache/data/c4")
    parser.add_argument("--output_dir", type=str, default="/n/home05/wk247/workspace/staged-training/local_cache/data/c4")
    # parser.add_argument("--input_dir", type=str, default='/n/tata_ddos_ceph/woojeong/data/c4')
    # parser.add_argument("--output_dir", type=str, default='/n/tata_ddos_ceph/woojeong/data/c4')

    parser.add_argument("--data_type", type=str, default="tfrecord")
    parser.add_argument("--add_sep_after_doc", action="store_true", default=False, help="add sep token after document")

    # Used only at the preprocessing phase
    parser.add_argument("--train_dev_split", type=float, default=0.05)
    parser.add_argument("--shard_size", type=int, default=1024 ** 3 // 4)  # 250MB
    parser.add_argument("--num_preprocessing_workers", type=int, default=1)
    # Used only at the training phase
    parser.add_argument("--seqlen", type=int, default=512)

    # HF model loading
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--doubling", type=str)  # could be layers / weights
    parser.add_argument("--doubling_layers", type=str)  # could be alternate_id, append_id,  alternate_copy, append_copy
    # parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--warmup_bsz", type=int, default=0, help="# warmup batch size")
    parser.add_argument("--warmup_seq", type=int, default=0, help="# warmup sequence length")

    parser.add_argument("--random", default=False, action="store_true")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--size", type=str)

    # Checkpointing and logging
    parser.add_argument("--save_dir", type=str, default="runs/")
    parser.add_argument("--save_prefix", type=str, default="test", help="path of output directory is --save_dir/--save_prefix")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,  # It is better to use a different output dir.
        help="Path to a checkpoint to load model weights and training state. It overwrites args",
    )
    parser.add_argument(
        "--resume_model_only", type=str, default=None, help="Path to a checkpoint to load model weights but not training state",
    )
    parser.add_argument("--reset_optimizer", default=False, action="store_true")
    parser.add_argument("--reset_lr_scheduler", default=False, action="store_true")
    parser.add_argument("--log_rate", type=int, default=10)
    parser.add_argument("--disable_checkpointing", action="store_true", default=False)

    # Training hyperparams
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train_steps", type=int, default=3000, help="# training grad. updates")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="# warmup grad. updates")
    parser.add_argument("--val_every", type=int, default=100, help="# training grad. updates between evaluations")
    parser.add_argument("--val_batches", type=int, default=1000, help="# evaluation **batches**")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--grad_clip", type=float, default=0)  # TODO: test this with fp16. Likely not working

    # RoBERTa's tokens_per_step = 2^18 = 512(seqlen) x 1(gpu_count) x 32(batch_size) x 16(grad_accum)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)

    # Compute resources
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--gpu_count",
        type=int,
        default=1,  # `--gpus` is reserved for internal use by PTL
        help="Number of gpus. This respects `CUDA_VISIBLE_DEVICES`",
    )

    # For restarting with warmup
    parser.add_argument("--restart_warmup_steps", type=int, default=0, help="# warmup grad. updates after restart")
    parser.add_argument("--restart_steps", type=int, default=0, help="# restart steps, should be the same as set_global_steps")
    # For multi-node training, use the PyTorch launch script. The script and instructions can be found here:
    # https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py.
    # To run PTL in a mode compatible with the launch script, two things are needed:
    #   - pass the argument `--use_env` to `torch.distributed.launch`
    #   - make sure `--nproc_per_node` matches `--gpu_count` and `--nnodes` matches `--node_count`.
    # For example, to run on 2 nodes, 3 gpus each, the command line on node rank 1 would be like:
    #   >>>> python -m torch.distributed.launch  \
    #               --use_env  --nnodes 2  --nproc_per_node 3  \
    #               --node_rank 1  --master_addr s2-server4  --master_port 12343  \
    #               scripts/pretrain.py  \
    #               --gpu_count 2  --node_count 2  \
    #               --input_dir my_data_dir  --save_prefix test_multinode
    parser.add_argument(
        "--node_count", type=int, default=1, help="Number of nodes. It needs to match --nnodes of torch.distributed.launch"
    )
    parser.add_argument("--tpu_core_count", type=int, default=None)

    return parser.parse_args()
