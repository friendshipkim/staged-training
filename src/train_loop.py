import copy
import torch
import numpy as np
from pytorch_lightning.trainer.training_loop import TrainLoop


# rewrite the MyTrainLoop from pytorch-lightning to support batch size and sequence length warmup
class MyTrainLoop(TrainLoop):
    def __init__(self, trainer, multiple_trainloader_mode, args):
        super().__init__(trainer, multiple_trainloader_mode)
        self.args = args

    def grad_norm(self, model, norm_type, should_accumulate=False):
        # Override PTL `grad_norm` function to only return `total_grad_norm` instead norms of individual params
        # TODO: grad_norm reporting needs to take fp16 loss scale into account
        # parameters = [p for p in self.parameters() if p.grad is not None]
        # device = parameters[0].device
        # total_norm = torch.zeros([], device=device if parameters else None)
        # norm_type = float(norm_type)
        # for p in parameters:
        #     param_norm = p.grad.norm(norm_type)
        #     total_norm.add_(param_norm)
        norm_type = float(norm_type)

        norms, all_norms = {}, []
        # local_norm = torch.zeros([], device=model.device)
        for name, p in model.named_parameters():
            if p.grad is None:
                continue

            if not should_accumulate:
                # param_norm = float(p.grad.data.norm(norm_type))
                p_grad = p.grad.data / self.args.batch_size / self.args.grad_accum
                param_norm = float(p_grad.norm(norm_type))
            else:
                p_grad = p.grad.data / self.trainer.accelerator.precision_plugin.scaler.get_scale() / self.args.batch_size
                param_norm = float(p_grad.norm(norm_type))
            all_norms.append(param_norm)
            # local_norm.add_(p.grad.norm(norm_type))

        total_norm = float(torch.tensor(all_norms).norm(norm_type))
        # norms[f'grad_{norm_type}_norm_total'] = round(total_norm, 4)
        # print("total_norm", total_norm, model.device, local_norm, self.trainer.accelerator.precision_plugin.scaler.get_scale())
        if not should_accumulate:
            return {
                "total_grad_norm": total_norm,
                "batch_size": self.args.batch_size * self.trainer.world_size,
                "grad_accum": self.args.grad_accum,
            }
        else:
            return {
                "local_grad_norm %s" % model.device: total_norm,
                "local_scale": self.trainer.accelerator.precision_plugin.scaler.get_scale(),
            }

    def _track_gradient_norm(self):
        grad_norm_dict = {}
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            if float(self.trainer.track_grad_norm) > 0:
                model = self.trainer.lightning_module
                grad_norm_dict = self.grad_norm(model, self.trainer.track_grad_norm)
        return grad_norm_dict

    def backward(self, result, optimizer, opt_idx, *args, **kwargs):
        self.trainer.dev_debugger.track_event("backward_call")

        should_accumulate = self.should_accumulate()
        # print(should_accumulate)
        # backward can be called manually in the training loop
        if isinstance(result, torch.Tensor):
            self.trainer.accelerator.backward(result, optimizer, opt_idx, should_accumulate, *args, **kwargs)
        else:
            result.closure_loss = self.trainer.accelerator.backward(
                result.closure_loss, optimizer, opt_idx, should_accumulate, *args, **kwargs
            )

        if not self.should_accumulate():
            # track gradients
            # print("track gradient with should_accumulate False")
            cur_grad_norm_dict = self.track_and_norm_grad(optimizer=optimizer)
            if "total_grad_norm" in self._cur_grad_norm_dict:
                B_small, B_big = (
                    self._cur_grad_norm_dict["batch_size"],
                    self._cur_grad_norm_dict["batch_size"] * self._cur_grad_norm_dict["grad_accum"],
                )
                grad_norm_B_big = self._cur_grad_norm_dict["total_grad_norm"]
                grad_norm_B_small = []
                if not hasattr(self, "grad_norm_dict") or (hasattr(self, "grad_norm_dict") and self.grad_norm_dict is None):
                    B_critical = B_big
                else:
                    for item in self.grad_norm_dict:
                        if "local_grad_norm" in item:
                            grad_norm_B_small.append(self.grad_norm_dict[item])

                    grad_norm_B_small = np.average(grad_norm_B_small)
                    g2 = 1 / (B_big - B_small) * (B_big * grad_norm_B_big - B_small * grad_norm_B_small)
                    s = 1 / (1 / B_small - 1 / B_big) * (grad_norm_B_small - grad_norm_B_big)
                    B_critical = s / g2
                    self._cur_grad_norm_dict.update(self.grad_norm_dict)
                self._cur_grad_norm_dict.update({"critical_batch_size": B_critical})
                for e in ["batch_size", "grad_accum"]:
                    self._cur_grad_norm_dict.pop(e)
                # print(self._cur_grad_norm_dict)
            self.grad_norm_dict = None
        else:
            # print("track gradient with should_accumulate True")
            # first gradient accumulation step !!!!!!!!!!!!
            if hasattr(self, "grad_norm_dict") and self.grad_norm_dict is None:
                model = self.trainer.lightning_module
                self.grad_norm_dict = self.grad_norm(model, self.trainer.track_grad_norm, True)

    def run_training_epoch(self):
        # modify dataloader if needed (ddp, etc...)
        train_dataloader = self.trainer.accelerator.process_dataloader(self.trainer.train_dataloader)

        # track epoch output
        epoch_output = [[] for _ in range(self.num_optimizers)]

        train_dataloader = self.trainer.data_connector.get_profiled_train_dataloader(train_dataloader)
        dataloader_idx = 0

        batch_idx = None
        is_last_batch = None

        accum_bsz, accum_bsz_grad_step = 0, 0
        for batch_idx, (batch, is_last_batch) in train_dataloader:
            self.trainer.batch_idx = batch_idx
            self.trainer.is_last_batch = is_last_batch

            # warmup the batch size via truncation and gradient accumulation
            # hack the deepest into the PTL to make it happen
            if self.args.warmup_bsz != 0:
                # for key in batch.keys():
                #     print(key, batch[key].shape, batch[key].device, batch[key].numel(), self.trainer.accumulate_grad_batches, self.trainer.model.device)
                input_ids = batch["input_ids"]

                final_bsz = input_ids.shape[0] * self.args.grad_accum * self.trainer.world_size
                start_bsz = 64

                current_bsz = start_bsz + (final_bsz - start_bsz) * min(1.0, accum_bsz / self.args.warmup_bsz)
                # print("before current_bsz", current_bsz, accum_bsz)
                if current_bsz >= final_bsz:
                    self.trainer.accumulate_grad_batches = self.args.grad_accum
                else:
                    current_bsz = current_bsz // self.trainer.world_size
                    # try to reset gradient accum steps
                    grad_accum = int(max(1, current_bsz // input_ids.shape[0]))

                    if grad_accum == 1 or accum_bsz_grad_step <= 0:
                        if grad_accum != 1 and accum_bsz_grad_step == 0:
                            accum_bsz_grad_step = grad_accum
                        self.trainer.accumulate_grad_batches = grad_accum
                        bsz_after_chunk = int(current_bsz // self.trainer.accumulate_grad_batches)
                    else:
                        accum_bsz_grad_step -= 1

                    # try to chunk the inputs
                    # print("current_bsz", current_bsz, "grad_accum", grad_accum, self.trainer.accumulate_grad_batches, accum_bsz_grad_step, self.should_accumulate(), 'bsz_after_chunk', bsz_after_chunk, input_ids.shape[0])
                    if bsz_after_chunk < input_ids.shape[0]:
                        for key in batch.keys():
                            batch[key] = torch.narrow(batch[key], 0, 0, bsz_after_chunk)  # .to( self.trainer.model.device )

                accum_bsz += batch["input_ids"].numel()

            if self.args.warmup_seq != 0:

                input_ids = batch["input_ids"]

                start_seq = 64
                final_seq = input_ids.shape[1]

                current_seq = int(start_seq + (final_seq - start_seq) * min(1.0, accum_bsz / self.args.warmup_seq))
                if accum_bsz_grad_step <= 0:
                    accum_bsz_grad_step = self.trainer.accumulate_grad_batches
                else:
                    accum_bsz_grad_step -= 1

                if current_seq < final_seq:
                    for key in batch.keys():
                        batch[key] = torch.narrow(batch[key], 1, 0, current_seq)

                accum_bsz += batch["input_ids"].numel()

            # ------------------------------------
            # TRAINING_STEP + TRAINING_STEP_END
            # ------------------------------------
            with self.trainer.profiler.profile("run_training_batch"):
                batch_output = self.run_training_batch(batch, batch_idx, dataloader_idx)

            # when returning -1 from train_step, we end epoch early
            if batch_output.signal == -1:
                break

            # hook
            # TODO: add outputs to batches
            self.on_train_batch_end(
                epoch_output, batch_output.training_step_output_for_epoch_end, batch, batch_idx, dataloader_idx,
            )

            # -----------------------------------------
            # SAVE METRICS TO LOGGERS
            # -----------------------------------------
            self.trainer.logger_connector.log_train_step_metrics(batch_output)

            # -----------------------------------------
            # VALIDATE IF NEEDED
            # -----------------------------------------
            should_check_val = self._should_check_val_fx(batch_idx, is_last_batch)
            if should_check_val:
                self.trainer.validating = True
                self.trainer.run_evaluation()
                self.trainer.training = True

            # -----------------------------------------
            # SAVE LOGGERS (ie: Tensorboard, etc...)
            # -----------------------------------------
            self.save_loggers_on_train_batch_end()

            # update LR schedulers
            monitor_metrics = copy.deepcopy(self.trainer.logger_connector.callback_metrics)
            self.update_train_loop_lr_schedulers(monitor_metrics=monitor_metrics)
            self.trainer.checkpoint_connector.has_trained = True

            self.trainer.total_batch_idx += 1

            # max steps reached, end training
            if (
                self.trainer.max_steps is not None
                and self.trainer.max_steps <= self.trainer.global_step + 1
                and self._accumulated_batches_reached()
            ):
                break

            # end epoch early
            # stop when the flag is changed or we've gone past the amount
            # requested in the batches
            if self.trainer.should_stop:
                break

            # stop epoch if we limited the number of training batches
            if self._num_training_batches_reached(is_last_batch):
                break

            # progress global step according to grads progress
            self.increment_accumulated_grad_global_step()

        if batch_idx is None:
            # dataloader/iterator did not produce a batch
            return

        # handle epoch_output on epoch end
        self.on_train_epoch_end(epoch_output)

        # log epoch metrics
        self.trainer.logger_connector.log_train_epoch_end_metrics(epoch_output)

        should_check_val = self._should_check_val_fx(batch_idx, is_last_batch, on_epoch=True)
        should_skip_eval = self.trainer.evaluation_loop.should_skip_evaluation(self.trainer.num_val_batches)
        should_train_only = self.trainer.disable_validation or should_skip_eval

        # update epoch level lr_schedulers if no val loop outside train loop is triggered
        if not should_check_val or should_train_only:
            self.trainer.optimizer_connector.update_learning_rates(interval="epoch")

        if should_train_only:
            self.check_checkpoint_callback(True)

        if should_check_val:
            self.trainer.validating = True
            self.trainer.run_evaluation(on_epoch=True)
            self.trainer.training = True

        if batch_output.signal != -1:
            self.increment_accumulated_grad_global_step()
