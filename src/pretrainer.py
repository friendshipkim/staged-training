import os
import time
import math
import torch
import pytorch_lightning as ptl
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from .dataset import MMapTextDataset
from .utils import get_restart_linear_schedule_with_warmup

import logging

logger = logging.getLogger(__name__)


try:
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


class Pretrainer(ptl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args  # hparams
        self._set_hparams(self.args)  # v1.3.5 ptl issue
        # self.hparams = self.args

        # self.model = AutoModelForMaskedLM.from_pretrained(args.model)
        if args.use_local_cache:
            local_cache_path = f"/n/home05/wk247/workspace/staged-training/local_cache/{args.tokenizer}"
            assert os.path.exists(local_cache_path), f"{local_cache_path} doesn't exist"
            self.model = AutoModelForCausalLM.from_pretrained(local_cache_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(args.model)
        if args.random:
            if args.layers is not None and args.size is not None:
                raise False
            if args.layers is not None:
                self.model.config.n_layer = args.layers
            if args.size is not None:
                if args.size == "GPT2_base":
                    self.model.config.n_layer = 12
                    self.model.config.n_embd = 768
                    self.model.config.n_head = 8
                elif args.size == "GPT2_large":
                    self.model.config.n_layer = 24
                    self.model.config.n_embd = 1536
                    self.model.config.n_head = 16
                elif args.size == "GPT2_base_div2_width":
                    self.model.config.n_layer = 12
                    self.model.config.n_embd = 384
                    self.model.config.n_head = 4
                elif args.size == "GPT2_base_div2_depth":
                    self.model.config.n_layer = 6
                    self.model.config.n_embd = 768
                    self.model.config.n_head = 8

                elif args.size == "GPT2_large_div4_width":
                    self.model.config.n_layer = 24
                    self.model.config.n_embd = 384
                    self.model.config.n_head = 4

                elif args.size == "GPT2_large_div2_width":
                    self.model.config.n_layer = 24
                    self.model.config.n_embd = 768
                    self.model.config.n_head = 8
                elif args.size == "GPT2_large_div4_depth":
                    self.model.config.n_layer = 6
                    self.model.config.n_embd = 1536
                    self.model.config.n_head = 16
                elif args.size == "GPT2_large_div2_depth":
                    self.model.config.n_layer = 12
                    self.model.config.n_embd = 1536
                    self.model.config.n_head = 16
                else:
                    assert False

            assert self.model.config.n_positions == 1024
            self.model.config.n_positions = args.seqlen
            self.model = GPT2LMHeadModel(config=self.model.config)
        else:
            assert args.layers is None
            assert args.size is None

        # local cache
        if args.use_local_cache:
            local_cache_path = f"/n/home05/wk247/workspace/staged-training/local_cache/{args.tokenizer}"
            assert os.path.exists(local_cache_path), f"{local_cache_path} doesn't exist"
            tokenizer = AutoTokenizer.from_pretrained(local_cache_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id
        self.bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id

        logger.info(f"Creating dataset cache from dir {self.args.input_dir}. This could be slow the first time.")
        MMapTextDataset.raw_text_to_mmap(args)

        # TODO: add support for other objective functions (whole word masking, BART, Pegasus)
        # self.data_collator = DataCollatorForLanguageModeling(
        #     tokenizer=tokenizer, mlm=True, mlm_probability=self.args.mlm_prob
        # )
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        self.start_time = 0

    def to(self, *args, **kwargs):
        param_count_before_to = len(list(self.parameters()))
        super().to(*args, **kwargs)
        if self.trainer.on_tpu:
            # if self.trainer.use_tpu:
            # need to re-tie the weights after moving to XLA!
            self.model.tie_weights()
            if "roberta" in self.args.model or "longformer" in self.args.model:
                self.model.lm_head.bias = self.model.lm_head.decoder.bias
        param_count_after_to = len(list(self.parameters()))
        assert param_count_before_to == param_count_after_to

    def forward(self, inputs):
        # for MLM
        # get the padding mask - 1 for NOT masked, 0 for MASKED/PAD
        # attention_mask = (input_ids != self.pad_token_id).int()

        # output is loss, prediction_scores, hidden_states
        # output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # for LM
        output = self.model(**inputs)
        return output[0]  # loss

    def training_step(self, batch, batch_nb):
        loss = self(batch)
        input_ids = batch["input_ids"]
        tensorboard_logs = {
            "input_size": input_ids.numel(),
            "token_per_step": input_ids.numel() * self.trainer.accumulate_grad_batches * self.trainer.world_size,
        }
        # if not self.use_tpu:
        if not self.trainer.on_tpu:
            # logging additional losses is slow on tpu
            tensorboard_logs["lm_loss"] = loss
            tensorboard_logs["lm_bpc"] = loss / math.log(2)
            tensorboard_logs["lm_perplexity"] = torch.exp(loss)

        if self.start_time != 0:
            # torch.cuda.synchronize()
            elapsed_time = time.monotonic() - self.start_time
            tensorboard_logs["second_per_batch"] = elapsed_time
        self.start_time = time.monotonic()

        if self.on_gpu:
            tensorboard_logs["memory"] = torch.cuda.memory_allocated(loss.device) / 1024 ** 3

        for k, v in tensorboard_logs.items():
            self.log(k, v)

        return {"loss": loss}

    def on_train_batch_start(self, *args, **kwargs):
        self._start = time.monotonic()

    def on_train_batch_end(self, *args, **kwargs):
        delta = time.monotonic() - self._start
        self.log("time_per_batch", delta, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_nb):
        # TODO: log how long evaluation takes
        self.start_time = 0  # reset training_step timer

        loss = self(batch)
        tensorboard_logs = {
            "val_lm_loss": loss.detach(),
        }
        return {"val_loss": tensorboard_logs["val_lm_loss"], "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["log"]["val_lm_loss"] for x in outputs if "val_lm_loss" in x["log"]]).mean()
        if self.trainer.accelerator_connector.use_ddp:
            # TODO: PTL is already doing this. Is it still needed here?
            # https://github.com/PyTorchLightning/pytorch-lightning/blob/0.8.5/pytorch_lightning/metrics/converters.py#L251
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= torch.distributed.get_world_size()
        elif self.on_tpu:
            avg_loss = xm.all_reduce(xm.REDUCE_SUM, avg_loss) / xm.xrt_world_size()

        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        # no_decay = ["bias", "LayerNorm.weight"]

        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
        #         "weight_decay": self.args.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
        #         "weight_decay": 0.0,
        #     },
        # ]
        # optimizer_grouped_parameters

        optimizer = AdamW(
            self.parameters(),
            lr=self.args.lr,
            eps=self.args.adam_epsilon,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            correct_bias=False,
        )
        if self.args.restart_warmup_steps != 0 and self.args.restart_steps != 0:
            scheduler = get_restart_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.train_steps,
                restart_steps=self.args.restart_steps,
                restart_warmup_steps=self.args.restart_warmup_steps,
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.train_steps
            )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_loader(self, fname, is_train):
        dataset = MMapTextDataset(
            fname, chunk_size=self.args.seqlen, bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id
        )

        # TODO: consider `replace_sampler_ddp=True` and removing the following if statement
        # if self.trainer.use_ddp:
        if self.trainer.accelerator_connector.use_ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
            shuffle = False
        elif self.trainer.on_tpu:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=is_train,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = is_train

        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.args.num_workers,
            collate_fn=self.data_collator,
            drop_last=is_train,
        )
        return loader

    def train_dataloader(self):
        return self._get_loader(f"{self.args.input_dir}/cache/train.bin", True)

    def val_dataloader(self):
        return self._get_loader(f"{self.args.input_dir}/cache/val.bin", False)

    def grad_norm(self, norm_type):
        # Override PTL `grad_norm` function to only return `total_grad_norm` instead norms of individual params
        # TODO: grad_norm reporting needs to take fp16 loss scale into account
        parameters = [p for p in self.parameters() if p.grad is not None]
        device = parameters[0].device
        total_norm = torch.zeros([], device=device if parameters else None)
        norm_type = float(norm_type)
        for p in parameters:
            param_norm = p.grad.norm(norm_type)
            total_norm.add_(param_norm)
        return {"total_grad_norm": total_norm}
