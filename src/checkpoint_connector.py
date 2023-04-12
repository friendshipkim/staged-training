import torch
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from pytorch_lightning.utilities import AMPType, rank_zero_warn
from pytorch_lightning.utilities.upgrade_checkpoint import KEYS_MAPPING as DEPRECATED_CHECKPOINT_KEYS
from pytorch_lightning.utilities.exceptions import MisconfigurationException

try:
    from apex import amp
except ImportError:
    amp = None


class MyCheckpointConnector(CheckpointConnector):
    def __init__(self, trainer, reset_optimizer=False, reset_lr_scheduler=False, set_global_step=None):
        super().__init__(trainer)
        self.reset_optimizer = reset_optimizer
        self.reset_lr_scheduler = reset_lr_scheduler
        self.set_global_step = set_global_step

    def restore_training_state(self, checkpoint, load_optimizer_states: bool = True):
        """
        COPIED from https://github.com/PyTorchLightning/pytorch-lightning/blob/1.0.8/pytorch_lightning/trainer/connectors/checkpoint_connector.py#L130-L199
        and updated to support reset_optimizer and reset_lr_scheduler
        """
        # validation
        if "optimizer_states" not in checkpoint or "lr_schedulers" not in checkpoint:
            raise KeyError(
                "Trying to restore training state but checkpoint contains only the model."
                " This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`."
            )

        if any([key in checkpoint for key in DEPRECATED_CHECKPOINT_KEYS]):
            raise ValueError(
                "The checkpoint you're attempting to load follows an"
                " outdated schema. You can upgrade to the current schema by running"
                " `python -m pytorch_lightning.utilities.upgrade_checkpoint --file model.ckpt`"
                " where `model.ckpt` is your checkpoint file."
            )

        # restore amp scaling
        if self.trainer.amp_backend == AMPType.NATIVE and "native_amp_scaling_state" in checkpoint:
            self.trainer.scaler.load_state_dict(checkpoint["native_amp_scaling_state"])
        elif self.trainer.amp_backend == AMPType.APEX and "amp_scaling_state" in checkpoint:
            amp.load_state_dict(checkpoint["amp_scaling_state"])

        # restore callback states
        self.trainer.on_load_checkpoint(checkpoint)

        self.trainer.global_step = checkpoint["global_step"]
        if self.set_global_step is not None:
            self.trainer.global_step = self.set_global_step
        self.trainer.current_epoch = checkpoint["epoch"]

        # crash if max_epochs is lower then the current epoch from the checkpoint
        if self.trainer.current_epoch > self.trainer.max_epochs:
            m = f"""
            you restored a checkpoint with current_epoch={self.trainer.current_epoch}
            but the Trainer(max_epochs={self.trainer.max_epochs})
            """
            raise MisconfigurationException(m)

        # Division deals with global step stepping once per accumulated batch
        # Inequality deals with different global step for odd vs even num_training_batches
        n_accum = 1 if self.trainer.accumulate_grad_batches is None else self.trainer.accumulate_grad_batches
        expected_steps = self.trainer.num_training_batches / n_accum
        if self.trainer.num_training_batches != 0 and self.trainer.global_step % expected_steps > 1:
            rank_zero_warn(
                "You're resuming from a checkpoint that ended mid-epoch. "
                "This can cause unreliable results if further training is done, "
                "consider using an end of epoch checkpoint. "
            )

        if not load_optimizer_states:
            return

        # restore the optimizers
        if not self.reset_optimizer:
            optimizer_states = checkpoint["optimizer_states"]
            for optimizer, opt_state in zip(self.trainer.optimizers, optimizer_states):
                print(opt_state.keys(), optimizer)
                # print(optimizer.param_groups.keys(), optimizer.param_groups)
                print([x.keys() for x in optimizer.param_groups])
                print([x.keys() for x in opt_state["param_groups"]])
                optimizer.load_state_dict(opt_state)

                # move optimizer to GPU 1 weight at a time
                # avoids OOM
                if self.trainer.root_gpu is not None:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda(self.trainer.root_gpu)

        if not self.reset_lr_scheduler:
            # restore the lr schedulers
            lr_schedulers = checkpoint["lr_schedulers"]
            if self.set_global_step is not None:
                for lrs_state in lr_schedulers:
                    lrs_state["last_epoch"] = self.set_global_step
                    lrs_state["_step_count"] = self.set_global_step + 1

            for scheduler, lrs_state in zip(self.trainer.lr_schedulers, lr_schedulers):
                scheduler["scheduler"].load_state_dict(lrs_state)
        else:
            if self.set_global_step is not None:
                for scheduler in self.trainer.lr_schedulers:
                    scheduler["scheduler"].last_epoch = self.set_global_step
                    scheduler["scheduler"]._step_count = self.set_global_step + 1
