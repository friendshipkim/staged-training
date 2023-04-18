import os
import re
import random
import copy
import logging
import torch
import numpy as np
import pytorch_lightning as ptl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.pretrainer import Pretrainer
from src.checkpoint_connector import MyCheckpointConnector
from src.train_loop import MyTrainLoop
from src.arguments import create_argument_parser
from src.operators import double_state_dict, double_param, deep_state_dict, deep_param


# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def grow_model(args, pretrainer):
    doubled_resume = args.resume + ".doubled_weights" if args.doubling == "weights" else args.resume + ".doubled_layer"
    print(doubled_resume)
    exist_flag = os.path.isfile(doubled_resume)

    if exist_flag:
        args.resume = doubled_resume
        print("================== warning: reusing old ckpt =======================")

    # doubling the checkpoint before doubling the in-memory model
    if args.resume is not None and not exist_flag:
        ckpt = torch.load(args.resume)

        # doubling state dict of the saved model
        if args.doubling == "weights":
            model_state_dict = ckpt["state_dict"]
            ckpt["state_dict"] = double_state_dict(model_state_dict, is_double_embedding=True)

            # doubling state dict of the saved optimizer
            # no_decay = ["bias", "LayerNorm.weight"]
            # optimizer_params_by_name = [(n, p.shape) for n, p in pretrainer.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad]
            # optimizer_params_by_name.extend([(n, p.shape) for n, p in pretrainer.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad])
            optimizer_params_by_name = [(n, p.shape) for n, p in pretrainer.named_parameters()]
            assert len(optimizer_params_by_name) == len(ckpt["optimizer_states"][0]["state"])
            for (param_name, param_shape), param in zip(optimizer_params_by_name, ckpt["optimizer_states"][0]["state"].values()):
                assert param["exp_avg"].shape == param_shape
                assert param["exp_avg_sq"].shape == param_shape
                param["exp_avg"] = double_param(
                    param_name, param["exp_avg"], is_double_embedding=True, is_grad=True, is_avg_sq=False
                )
                param["exp_avg_sq"] = double_param(
                    param_name, param["exp_avg_sq"], is_double_embedding=True, is_grad=True, is_avg_sq=True
                )

                # print(name_shape[0])
            args.resume += ".doubled_weights"
        elif args.doubling == "layers":
            model_state_dict = ckpt["state_dict"]
            # hack for doubling the layers
            prefix = "model.transformer.h"
            map_positions, copy_positions = {}, {}
            for key in model_state_dict:
                if prefix in key:
                    layer_idx = re.findall("[-\d]+", key)[0]
                    origin_idx = prefix + "." + str(int(layer_idx))
                    if "alternate" in args.doubling_layers:
                        insert_idx = prefix + "." + str(int(layer_idx) * 2 + 1)
                        origin_key = key.replace(origin_idx, prefix + "." + str(int(layer_idx) * 2))
                    elif "append" in args.doubling_layers:
                        insert_idx = prefix + "." + str(pretrainer.model.config.n_layer + int(layer_idx))
                        origin_key = key

                    insert_key = key.replace(origin_idx, insert_idx)

                    map_positions[key] = [(origin_key, False), (insert_key, False)]
                    copy_positions[insert_key] = (key, False)
                    copy_positions[origin_key] = (key, True)

            is_identical = "id" in args.doubling_layers

            ckpt["state_dict"] = deep_state_dict(model_state_dict, is_identical=is_identical, map_positions=map_positions)

            # deal with the optimizer state
            original_optimizer_params_by_name = [(n, p.shape) for n, p in pretrainer.named_parameters()]
            # print( "original_optimizer_params_by_name", original_optimizer_params_by_name )
            # print( "ckpt optimizer_states", ckpt['optimizer_states'][0]['state'].keys() )
            layers = pretrainer.model.transformer.h
            n = len(layers)
            for i in range(n):
                if "alternate" in args.doubling_layers:
                    layers.insert(i * 2, copy.deepcopy(layers[i * 2]))
                elif "append" in args.doubling_layers:
                    layers.append(copy.deepcopy(layers[i]))

            pretrainer.model.config.n_layer *= 2
            pretrainer.model.tie_weights()
            new_optimizer_params_by_name = [(n, p.shape) for n, p in pretrainer.named_parameters()]

            new_optimizer_state = {_: {} for _ in range(len(new_optimizer_params_by_name))}
            assert len(original_optimizer_params_by_name) == len(ckpt["optimizer_states"][0]["state"])
            original_optimizer_param_name_dict = {}
            for (param_name, param_shape), param in zip(
                original_optimizer_params_by_name, ckpt["optimizer_states"][0]["state"].values()
            ):
                assert param["exp_avg"].shape == param_shape
                assert param["exp_avg_sq"].shape == param_shape
                original_optimizer_param_name_dict[param_name] = copy.deepcopy(param)

            for param_idx, (param_name, param_shape) in enumerate(new_optimizer_params_by_name):
                if copy_positions.get(param_name):
                    copy_param_name, copy_param_flag = copy_positions.get(param_name)
                    param_is_identical = copy_param_flag and is_identical
                    new_optimizer_state[param_idx] = copy.deepcopy(original_optimizer_param_name_dict[copy_param_name])
                    new_optimizer_state[param_idx]["exp_avg"] = deep_param(
                        param_name,
                        original_optimizer_param_name_dict[copy_param_name]["exp_avg"],
                        is_identical=param_is_identical,
                        is_grad=True,
                        is_avg_sq=False,
                    )
                    new_optimizer_state[param_idx]["exp_avg_sq"] = deep_param(
                        param_name,
                        original_optimizer_param_name_dict[copy_param_name]["exp_avg_sq"],
                        is_identical=param_is_identical,
                        is_grad=True,
                        is_avg_sq=True,
                    )
                else:
                    new_optimizer_state[param_idx] = copy.deepcopy(original_optimizer_param_name_dict[param_name])

            ckpt["optimizer_states"][0]["state"] = new_optimizer_state
            ckpt["optimizer_states"][0]["param_groups"][0]["params"] = list(new_optimizer_state.keys())
            del original_optimizer_param_name_dict
            args.resume += ".doubled_layer"

        torch.save(ckpt, args.resume)
        exit()

    # we need to resume the model after the doubling
    if args.doubling == "layers":
        assert True
    elif args.doubling == "weights":
        assert True
    else:
        assert False


def main(args):
    random.seed(args.seed * 10)
    np.random.seed(args.seed * 100)
    torch.manual_seed(args.seed * 1000)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed * 10000)

    # initialize pretrainer
    if args.resume_model_only is not None:
        pretrainer = Pretrainer.load_from_checkpoint(args.resume_model_only)
    else:
        pretrainer = Pretrainer(args)

    # grow the model
    if args.doubling is not None:
        grow_model(args, pretrainer)

    # logger here is a SummaryWritter for tensorboard
    # it is used by the trainer, and certain return variables
    # from the model are automatically logged
    # logger = TestTubeLogger(save_dir=args.save_dir, name=args.save_prefix, version=0)  # always use version=0
    wandb_logger = WandbLogger(project="gpt2-pretraining", name=f"{args.save_prefix}",)

    # log gradients and learning rate
    wandb_logger.watch(pretrainer)
    # TODO: init wandb only once, maybe turn off watch?
    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        # model saved to filepath/prefix_....
        # filepath=os.path.join(args.save_dir, args.save_prefix, 'checkpoint'),
        # prefix='',
        dirpath=os.path.join(args.save_dir, args.save_prefix),
        filename="checkpoint-{epoch}-{step}",
        save_top_k=-1,
        # save_top_k=10,
        every_n_train_steps=250,
        save_last=True,
        verbose=True,
        # monitor='val_loss',
        # mode='min',
    )
    
    # PTL is expecting number of batches_per_gpu
    args.val_every *= args.grad_accum
    
    logger.info(f"Validate every {args.val_every} steps")
    logger.info(f"Checkpointing?: {not args.disable_checkpointing}")
    logger.info(checkpoint_callback.__dict__)
    
    trainer = ptl.Trainer(
        gpus=args.gpu_count,
        num_nodes=args.node_count,
        tpu_cores=args.tpu_core_count,
        distributed_backend="ddp",  # if (args.gpu_count > 1 or args.node_count > 1) else None,
        replace_sampler_ddp=False,
        track_grad_norm=2 if args.tpu_core_count is None else -1,  # gradnorm logging is slow on tpus
        max_epochs=10000,
        min_epochs=0,
        max_steps=args.train_steps,  # run for many epochs, but stop after max_steps
        val_check_interval=args.val_every,
        limit_val_batches=args.val_batches,
        log_every_n_steps=args.log_rate,
        progress_bar_refresh_rate=args.log_rate,
        logger=wandb_logger,
        # checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else None,
        accumulate_grad_batches=args.grad_accum,
        resume_from_checkpoint=args.resume,
        gradient_clip_val=args.grad_clip,
        precision=16 if args.fp16 else 32,
        amp_level="O2",
        num_sanity_val_steps=2,
        callbacks=[lr_monitor, checkpoint_callback],
        profiler="simple",
    )
    trainer.profiler.dirpath = os.path.join(args.save_dir, args.save_prefix)
    trainer.profiler.filename = "profiler"
    trainer.train_loop = MyTrainLoop(trainer, multiple_trainloader_mode="max_size_cycle", args=args)
    trainer.checkpoint_connector = MyCheckpointConnector(
        trainer,
        reset_lr_scheduler=args.reset_lr_scheduler,
        reset_optimizer=args.reset_optimizer,
        set_global_step=args.restart_steps + 1,
    )
    trainer.fit(pretrainer)


if __name__ == "__main__":
    args = create_argument_parser()
    main(args)
