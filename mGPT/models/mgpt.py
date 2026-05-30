import numpy as np
import os
import random
import torch
import time
from mGPT.archs.root_correction import RootCorrectionHead
from mGPT.config import get_obj_from_str, instantiate_from_config
from mGPT.data.humanml.scripts.motion_process import recover_root_rot_pos
from os.path import join as pjoin
from mGPT.losses.mgpt import GPTLosses
from mGPT.models.base import BaseModel
from .base import BaseModel
import json
import mGPT.render.matplot.plot_3d_global as plot_3d


class MotionGPT(BaseModel):
    """
    Stage 1 Motion Tokenizer
    Stage 2 Motion-language pretrian
    Stage 3 Motion-language instruction tuning
    """

    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
                 codebook_size=512,
                 stage='vae',
                 debug=True,
                 condition='text',
                 task='t2m',
                 metrics_dict=['TM2TMetrics'],
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=False)
        self.datamodule = datamodule
        super().__init__()

        # Instantiate motion tokenizer
        if motion_vae != None:
            self.vae = instantiate_from_config(motion_vae)
        self._configure_root_velocity_calibration(cfg)
        self._configure_root_correction_head(cfg)

        # Instantiate the motion-language model only for LM stages.
        self.lm = instantiate_from_config(lm) if 'lm' in self.hparams.stage else None

        # Freeze the motion tokenizer for lm training
        if 'lm' in self.hparams.stage:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

        # Instantiate the losses
        self._losses = torch.nn.ModuleDict({
            split: GPTLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # Data transform
        self.feats2joints = datamodule.feats2joints

        # Count codebook frequency
        self.codePred = []
        self.codeFrequency = torch.zeros((self.hparams.codebook_size, ))

    def _configure_root_velocity_calibration(self, cfg):
        calib_cfg = cfg.get("ROOT_VEL_CALIB", {})
        self.root_velocity_calib_enabled = bool(calib_cfg.get("ENABLED", False))
        self.root_velocity_calib_domain = str(calib_cfg.get("DOMAIN", "m4human"))
        self.root_velocity_calib_freeze_vae = bool(calib_cfg.get("FREEZE_VAE", False))
        if not self.root_velocity_calib_enabled:
            return

        scale_min = float(calib_cfg.get("SCALE_MIN", 0.8))
        scale_max = float(calib_cfg.get("SCALE_MAX", 1.3))
        if not scale_min < 1.0 < scale_max:
            raise ValueError("ROOT_VEL_CALIB scale range must contain 1.0")
        init_scale = float(calib_cfg.get("INIT_SCALE", 1.0))
        init_scale = min(max(init_scale, scale_min + 1e-6), scale_max - 1e-6)
        init_prob = (init_scale - scale_min) / (scale_max - scale_min)
        init_logit = float(np.log(init_prob / (1.0 - init_prob)))

        self.register_buffer(
            "root_velocity_scale_bounds",
            torch.tensor([scale_min, scale_max], dtype=torch.float32),
        )
        self.root_velocity_scale_logit = torch.nn.Parameter(
            torch.full((2,), init_logit, dtype=torch.float32)
        )
        if bool(calib_cfg.get("AFFINE", False)):
            self.root_velocity_bias = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

        if self.root_velocity_calib_freeze_vae and hasattr(self, "vae"):
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False

    def _configure_root_correction_head(self, cfg):
        head_cfg = cfg.get("ROOT_CORRECTION_HEAD", {})
        self.root_correction_enabled = bool(head_cfg.get("ENABLED", False))
        self.root_correction_domain = str(head_cfg.get("DOMAIN", "m4human"))
        self.root_correction_freeze_vae = bool(head_cfg.get("FREEZE_VAE", False))
        if not self.root_correction_enabled:
            return

        self.root_correction_inputs = [
            str(item) for item in head_cfg.get("INPUTS", ["decoded"])
        ]
        input_dims = {
            "decoded": int(self.hparams.cfg.DATASET.NFEATS),
            "decoder_hidden": int(head_cfg.get(
                "DECODER_HIDDEN_DIM",
                self.hparams.cfg.model.params.motion_vae.params.get(
                    "width", 512))),
            "quantized": int(head_cfg.get(
                "QUANTIZED_DIM",
                self.hparams.cfg.model.params.motion_vae.params.get(
                    "code_dim", 512))),
        }
        unknown_inputs = [
            name for name in self.root_correction_inputs
            if name not in input_dims
        ]
        if unknown_inputs:
            raise ValueError(
                f"Unsupported ROOT_CORRECTION_HEAD.INPUTS={unknown_inputs}")
        correction_input_dim = sum(
            input_dims[name] for name in self.root_correction_inputs)

        self.root_correction_head = RootCorrectionHead(
            nfeats=correction_input_dim,
            hidden_dims=head_cfg.get("HIDDEN_DIMS", [256, 128]),
            kernel_size=int(head_cfg.get("KERNEL_SIZE", 3)),
            output_dim=3,
            zero_init=bool(head_cfg.get("ZERO_INIT", True)),
        )

        if self.root_correction_freeze_vae and hasattr(self, "vae"):
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False

        self.root_correction_decoder_tail_modules = []
        tail_modules = int(head_cfg.get("UNFREEZE_DECODER_TAIL_MODULES", 0))
        if tail_modules > 0 and hasattr(self, "vae"):
            decoder_modules = list(self.vae.decoder.model.children())
            for module in decoder_modules[-tail_modules:]:
                self.root_correction_decoder_tail_modules.append(module)
                for param in module.parameters():
                    param.requires_grad = True
            if bool(head_cfg.get("ROOT_OUTPUT_ONLY", True)):
                self._register_root_output_grad_mask()

    def _register_root_output_grad_mask(self):
        if not hasattr(self, "vae"):
            return
        final = self.vae.decoder.model[-1]
        if not isinstance(final, torch.nn.Conv1d):
            return

        def mask_weight_grad(grad):
            masked = grad.clone()
            masked[3:] = 0.0
            return masked

        def mask_bias_grad(grad):
            masked = grad.clone()
            masked[3:] = 0.0
            return masked

        final.weight.register_hook(mask_weight_grad)
        if final.bias is not None:
            final.bias.register_hook(mask_bias_grad)

    def _root_velocity_scale(self):
        bounds = self.root_velocity_scale_bounds.to(self.root_velocity_scale_logit)
        scale_min, scale_max = bounds[0], bounds[1]
        return scale_min + (scale_max - scale_min) * torch.sigmoid(
            self.root_velocity_scale_logit
        )

    def _root_velocity_domain_mask(self, batch, device, batch_size):
        domain = getattr(self, "root_velocity_calib_domain", "m4human")
        if domain in ("all", "*"):
            return None
        if batch is None or "domain" not in batch:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        return torch.tensor(
            [item == domain for item in batch["domain"]],
            device=device,
            dtype=torch.bool,
        )

    def _apply_root_velocity_calibration(self, feats, batch=None):
        if not getattr(self, "root_velocity_calib_enabled", False):
            return feats
        if not hasattr(self, "root_velocity_scale_logit"):
            return feats
        mask = self._root_velocity_domain_mask(batch, feats.device, feats.shape[0])
        if mask is not None and not bool(mask.any()):
            return feats

        scale = self._root_velocity_scale().to(device=feats.device, dtype=feats.dtype)
        bias = getattr(self, "root_velocity_bias", None)
        if bias is None:
            bias = torch.zeros_like(scale)
        else:
            bias = bias.to(device=feats.device, dtype=feats.dtype)

        calibrated = feats.clone()
        if mask is None:
            calibrated[..., 1:3] = calibrated[..., 1:3] * scale + bias
        else:
            calibrated[mask, :, 1:3] = calibrated[mask, :, 1:3] * scale + bias
        return calibrated

    def _root_correction_domain_mask(self, batch, device, batch_size):
        domain = getattr(self, "root_correction_domain", "m4human")
        if domain in ("all", "*"):
            return None
        if batch is None or "domain" not in batch:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        return torch.tensor(
            [item == domain for item in batch["domain"]],
            device=device,
            dtype=torch.bool,
        )

    def _root_correction_input(self, feats, intermediates=None):
        inputs = []
        for name in getattr(self, "root_correction_inputs", ["decoded"]):
            if name == "decoded":
                inputs.append(feats)
                continue
            if intermediates is None or name not in intermediates:
                raise RuntimeError(
                    f"Root correction input '{name}' requires VQ-VAE "
                    "intermediates")
            inputs.append(intermediates[name])
        return torch.cat(inputs, dim=-1)

    def _apply_root_correction(self, feats, batch=None, intermediates=None):
        if not getattr(self, "root_correction_enabled", False):
            return feats
        if not hasattr(self, "root_correction_head"):
            return feats
        mask = self._root_correction_domain_mask(batch, feats.device, feats.shape[0])
        if mask is not None and not bool(mask.any()):
            return feats

        correction_input = self._root_correction_input(feats, intermediates)
        delta = self.root_correction_head(correction_input)
        corrected = feats.clone()
        if mask is None:
            corrected[..., :3] = corrected[..., :3] + delta
        else:
            corrected[mask, :, :3] = corrected[mask, :, :3] + delta[mask]
        return corrected

    def _vae_forward_for_root_correction(self, features):
        needs_intermediates = any(
            name != "decoded"
            for name in getattr(self, "root_correction_inputs", ["decoded"]))
        if getattr(self, "root_correction_enabled", False) and needs_intermediates:
            feats_rst, loss_commit, perplexity, intermediates = (
                self.vae.forward_with_intermediates(features))
            return feats_rst, loss_commit, perplexity, intermediates
        feats_rst, loss_commit, perplexity = self.vae(features)
        return feats_rst, loss_commit, perplexity, None

    def load_state_dict(self, state_dict, strict=True):
        if self.lm is None:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("lm.")
            }
        return super().load_state_dict(state_dict, strict=strict)

    def train(self, mode=True):
        super().train(mode)
        freeze_vae = (
            'lm' in self.hparams.stage or
            getattr(self, "root_velocity_calib_freeze_vae", False) or
            getattr(self, "root_correction_freeze_vae", False)
        )
        if mode and freeze_vae and hasattr(self, "vae"):
            self.vae.eval()
        return self

    def _module_grad_norm(self, module):
        total = None
        for param in module.parameters():
            if param.grad is None:
                continue
            value = param.grad.detach().norm(2).pow(2)
            total = value if total is None else total + value
        if total is None:
            return torch.tensor(0.0, device=self.device)
        return total.sqrt()

    def on_before_optimizer_step(self, optimizer):
        if not getattr(self, "root_correction_enabled", False):
            return
        if self.trainer.sanity_checking:
            return
        if hasattr(self, "root_correction_head"):
            self.log(
                "grad/root_correction_head",
                self._module_grad_norm(self.root_correction_head),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True,
            )
        modules = getattr(self, "root_correction_decoder_tail_modules", [])
        if modules:
            total = torch.tensor(0.0, device=self.device)
            for module in modules:
                total = total + self._module_grad_norm(module).pow(2)
            self.log(
                "grad/decoder_tail",
                total.sqrt(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True,
            )

    def configure_optimizers(self):
        if not getattr(self, "root_correction_enabled", False):
            return super().configure_optimizers()

        cfg = self.hparams.cfg
        head_cfg = cfg.get("ROOT_CORRECTION_HEAD", {})
        optim_target = cfg.TRAIN.OPTIM.target
        if len(optim_target.split('.')) == 1:
            optim_target = 'torch.optim.' + optim_target
        base_lr = float(cfg.TRAIN.OPTIM.params.lr)
        head_lr = float(head_cfg.get("HEAD_LR", base_lr))
        tail_lr = float(head_cfg.get("DECODER_TAIL_LR", base_lr))

        grouped_ids = set()
        param_groups = []
        head_params = [
            param for param in self.root_correction_head.parameters()
            if param.requires_grad
        ]
        if head_params:
            grouped_ids.update(id(param) for param in head_params)
            param_groups.append({"params": head_params, "lr": head_lr})

        tail_params = []
        for module in getattr(self, "root_correction_decoder_tail_modules", []):
            tail_params.extend([
                param for param in module.parameters()
                if param.requires_grad and id(param) not in grouped_ids
            ])
        if tail_params:
            grouped_ids.update(id(param) for param in tail_params)
            param_groups.append({"params": tail_params, "lr": tail_lr})

        other_params = [
            param for param in self.parameters()
            if param.requires_grad and id(param) not in grouped_ids
        ]
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr})

        optim_params = dict(cfg.TRAIN.OPTIM.params)
        optim_params.pop("lr", None)
        optimizer = get_obj_from_str(optim_target)(
            params=param_groups, **optim_params)

        scheduler_target = cfg.TRAIN.LR_SCHEDULER.target
        if len(scheduler_target.split('.')) == 1:
            scheduler_target = 'torch.optim.lr_scheduler.' + scheduler_target
        lr_scheduler = get_obj_from_str(scheduler_target)(
            optimizer=optimizer, **cfg.TRAIN.LR_SCHEDULER.params)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def forward(self, batch, task="t2m"):
        if self.lm is None:
            raise RuntimeError(
                f"Motion-language model is not initialized for stage "
                f"{self.hparams.stage}."
            )

        texts = batch["text"]
        lengths_ref = batch["length"]

        # Forward
        # texts = ['Generate motion: ' + text for text in texts]
        outputs, output_texts = self.lm.generate_direct(texts, do_sample=True)

        # Motion Decode
        feats_rst_lst = []
        lengths = []
        max_len = 0

        for i in range(len(texts)):
            if task == "pred":
                motion = self.vae.decode(
                    torch.cat((batch["motion"][i], outputs[i])))
            elif task in ["t2m", "m2t", "inbetween"]:
                motion = self.vae.decode(outputs[i])
                # motion = self.datamodule.denormalize(motion)
                lengths.append(motion.shape[1])
            else:
                raise NotImplementedError

            if motion.shape[1] > max_len:
                max_len = motion.shape[1]

            if task in ["t2m", "m2t", "pred"]:
                feats_rst_lst.append(motion)

            elif task == "inbetween":
                motion = torch.cat(
                    (batch["motion_heading"][i][None],
                     motion[:, lengths_ref[i] // 4:lengths_ref[i] // 4 * 3,
                            ...], batch["motion_tailing"][i][None]),
                    dim=1)
                feats_rst_lst.append(motion)

        feats_rst = torch.zeros(
            (len(feats_rst_lst), max_len, motion.shape[-1])).to(self.device)

        # padding and concat
        for i in range(len(feats_rst_lst)):
            feats_rst[i, :feats_rst_lst[i].shape[1], ...] = feats_rst_lst[i]

        # Recover joints for evaluation
        joints_rst = self.feats2joints(feats_rst)

        # return set
        outputs = {
            "texts": output_texts,
            "feats": feats_rst,
            "joints": joints_rst,
            "length": lengths
        }

        return outputs

    def train_lm_forward(self, batch):
        tokens_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch["tasks"]
        all_captions = batch['all_captions']
        if self.hparams.condition == 'caption':
            texts = [random.choice(all_captions[i]) for i in range(len(texts))]

        # LLM Forward
        outputs = self.lm(texts, tokens_ref, lengths, tasks)
        # outputs = self.t2m_gpt.generate(texts)
        return {'outputs': outputs}

    @torch.no_grad()
    def val_t2m_forward(self, batch):
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = None
        if self.trainer.datamodule.is_mm:
            texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            instructions = pjoin(self.datamodule.hparams.data_root,
                                 'template_instructions.json')
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["caption"]] * len(texts)

        if self.hparams.condition == 'caption':
            tasks = [{
                'input': ['<Caption_Placeholder>'],
                'output': ['']
            }] * len(texts)

        if self.hparams.cfg.DATASET.TASK_PATH:
            instructions = pjoin(self.hparams.cfg.DATASET.TASK_PATH)
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["t2m"]] * len(texts)

        min_len = lengths.copy()
        # Forward
        outputs = self.lm.generate_conditional(texts,
                                               lengths=lengths,
                                               stage='test',
                                               tasks=tasks)

        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)

        for i in range(len(texts)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)

            if len(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])

            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len
            # "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_m2t_forward(self, batch):
        self.hparams.metrics_dict = []

        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        all_captions = batch['all_captions']

        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
            motion_tokens.append(motion_token[0])
            lengths_tokens.append(motion_token.shape[1])

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths_tokens,
                                               task="m2t",
                                               stage='test')

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "t_ref": all_captions,
            # "t_ref": texts,
            "t_pred": outputs,
            "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_m2m_forward(self, batch, task="pred"):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
            motion_tokens.append(motion_token[0])

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths,
                                               task=task,
                                               stage='test')

        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)
        min_len = lengths.copy()

        for i in range(len(lengths)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)

            if len(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])

            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len
            # "length": lengths
        }

        return rs_set

    def train_vae_forward(self, batch):
        # batch detach
        feats_ref = batch["motion"]
        # motion encode & decode
        feats_rst, loss_commit, perplexity, intermediates = (
            self._vae_forward_for_root_correction(feats_ref))
        feats_rst = self._apply_root_velocity_calibration(feats_rst, batch)
        feats_rst = self._apply_root_correction(feats_rst, batch, intermediates)
        feats_ref_denorm = self.datamodule.denormalize(feats_ref)
        feats_rst_denorm = self.datamodule.denormalize(feats_rst)
        _, root_ref = recover_root_rot_pos(feats_ref_denorm)
        _, root_rst = recover_root_rot_pos(feats_rst_denorm)
        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "m_ref_denorm": feats_ref_denorm,
            "m_rst_denorm": feats_rst_denorm,
            "root_ref": root_ref,
            "root_rst": root_rst,
            "domain": batch.get("domain"),
            "loss_commit": loss_commit,
            "perplexity": perplexity,
        }
        return rs_set

    @torch.no_grad()
    def val_vae_forward(self, batch, split="train"):
        # Detach batch
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Repeat for multimodal evaluation
        if self.trainer.datamodule.is_mm:
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        # Motion encode & decode
        feats_rst = torch.zeros_like(feats_ref)

        for i in range(len(feats_ref)):
            if lengths[i] == 0:
                continue
            feats_pred, _, _, intermediates = self._vae_forward_for_root_correction(
                feats_ref[i:i + 1, :lengths[i]])
            sample_batch = None
            if "domain" in batch:
                sample_batch = {"domain": [batch["domain"][i]]}
            feats_pred = self._apply_root_velocity_calibration(
                feats_pred, sample_batch)
            if getattr(self, "root_correction_enabled", False):
                feats_pred = self._apply_root_correction(
                    feats_pred, sample_batch, intermediates)
            feats_rst[i:i + 1, :feats_pred.shape[1], :] = feats_pred

            # codeFre_pred = torch.bincount(code_pred[0],
            #                               minlength=self.hparams.codebook_size).to(
            #                                   self.codeFrequency.device)
            # self.codePred.append(code_pred[0])
            # self.codeFrequency += codeFre_pred

        # np.save('../memData/results/codeFrequency.npy',
        #         self.codeFrequency.cpu().numpy())

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # Return set
        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "length": lengths,
        }

        return rs_set


    def allsplit_step(self, split: str, batch, batch_idx):
        # Compute the losses
        loss = None

        if self.hparams.stage == "vae" and split in ["train", "val"]:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage in ["lm_instruct", "lm_pretrain"
                                    ] and split in ["train"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage == 'lm_rl' and split in ['train']:
            rs_set = self.train_rl_forward(batch)
            loss = None

        # Compute the metrics
        if split in ["val", "test"]:
            if self.hparams.stage == "vae":
                rs_set = self.val_vae_forward(batch, split)
            elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_rl"]:
                if self.hparams.task == "t2m":
                    rs_set = self.val_t2m_forward(batch)
                elif self.hparams.task == "m2t":
                    rs_set = self.val_m2t_forward(batch)
                elif self.hparams.task in ["m2m", "pred", "inbetween"]:
                    rs_set = self.val_m2m_forward(batch, self.hparams.task)

            if self.hparams.task not in ["m2t"]:
                # MultiModality evaluation sperately
                if self.trainer.datamodule.is_mm:
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.hparams.metrics_dict
                    
                if self.hparams.task not in ['pred', 'inbetween'] and 'PredMetrics' in metrics_dicts:
                    metrics_dicts.remove('PredMetrics')

                for metric in metrics_dicts:
                    lengths = batch['length']
                    if metric == "TemosMetric":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "TM2TMetrics":
                        if self.hparams.stage in [
                                "lm_instruct", "lm_pretrain", "lm_rl"
                        ]:
                            word_embs = batch['word_embs']
                            pos_ohot = batch['pos_ohot']
                            text_lengths = batch['text_len']
                            if self.trainer.datamodule.is_mm:
                                word_embs = word_embs.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                                pos_ohot = pos_ohot.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                                text_lengths = text_lengths.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                        else:
                            word_embs = None
                            pos_ohot = None
                            text_lengths = None

                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            feats_rst=rs_set["m_rst"],
                            lengths_ref=lengths,
                            lengths_rst=rs_set['length'],
                            word_embs=word_embs,
                            pos_ohot=pos_ohot,
                            text_lengths=text_lengths,
                        )
                    elif metric == "UncondMetrics":
                        getattr(self.metrics, metric).update(
                            recmotion_embeddings=rs_set["lat_rm"],
                            gtmotion_embeddings=rs_set["lat_m"],
                            lengths=lengths,
                        )
                    elif metric == "MRMetrics":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "PredMetrics":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "MMMetrics":
                        # pass
                        getattr(self.metrics,
                                metric).update(rs_set["m_rst"],
                                               rs_set['length'])
                    else:
                        raise TypeError(f"Not support this metric {metric}")

            elif self.hparams.task == "m2t" and self.hparams.stage in [
                    "lm_instruct", "lm_pretrain", "lm_rl"
            ]:
                self.hparams.metrics_dict = metrics_dicts = ['M2TMetrics']
                for metric in metrics_dicts:
                    if metric == "M2TMetrics":
                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            pred_texts=rs_set["t_pred"],
                            gt_texts=batch["all_captions"],
                            lengths=rs_set['length'],
                            word_embs=batch["word_embs"],
                            pos_ohot=batch["pos_ohot"],
                            text_lengths=batch["text_len"],
                        )

        # return forward output rather than loss during test
        if split in ["test"]:
            if self.hparams.task == "t2m":
                return rs_set["joints_rst"], rs_set["length"], rs_set[
                    "joints_ref"]
                # pass
            elif self.hparams.task == "m2t":
                return rs_set["t_pred"], batch["length"]
                # return batch["length"]

        return loss
