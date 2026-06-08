import os

import ocnn
import torch as th
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.functional.classification import (
    binary_f1_score,
    binary_accuracy,
    binary_recall,
    binary_precision,
)
from torchvision.ops import sigmoid_focal_loss

import webdataset as wds

from zerograsp.utils.config import parse_config
from zerograsp.utils.dataset import make_sample_wrapper, decode_depth
from zerograsp.nets import ZeroGrasp

th.set_float32_matmul_precision("medium")

INTRINSICS_K = {
    "mirage": [
        [572.41136339, 0.0, 325.2611084],
        [0.0, 573.57043286, 242.04899588],
        [0.0, 0.0, 1.0],
    ],
    "graspnet": [
        [927.1697387695312, 0.0, 651.3150634765625],
        [0.0, 927.3668823242188, 349.621337890625],
        [0.0, 0.0, 1.0],
    ],
    "ycb_video": [
        [1066.778, 0.0, 312.9869],
        [0.0, 1067.487, 241.3109],
        [0.0, 0.0, 1.0],
    ],
    "hope": [[1390.53, 0.0, 964.957], [0.0, 1386.99, 522.586], [0.0, 0.0, 1.0]],
    "hb": [[537.4799, 0.0, 318.8965], [0.0, 536.1447, 238.3781], [0.0, 0.0, 1.0]],
    "woven_easy": [
        [610.1778394083658, 0.0, 640.0],
        [0.0, 610.1778394082355, 512.0],
        [0.0, 0.0, 1.0],
    ],
    "woven_normal": [
        [610.1778394083658, 0.0, 640.0],
        [0.0, 610.1778394082355, 512.0],
        [0.0, 0.0, 1.0],
    ],
    "woven_hard": [
        [610.1778394083658, 0.0, 640.0],
        [0.0, 610.1778394082355, 512.0],
        [0.0, 0.0, 1.0],
    ],
    "maniskill": [
        [512.0, 0.0, 640.0],
        [0.0, 512.0, 512.0],
        [0.0, 0.0, 1.0],
    ],
}


def _remap_legacy_checkpoint_key(key: str) -> str:
    replacements = (
        (".wrap_att.layer.norm.", ".attn_norm.norm."),
        (".wrap_att.layer.sublayer.qkv_proj.", ".attn.qkv_proj."),
        (".wrap_att.layer.sublayer.q_proj.", ".attn.q_proj."),
        (".wrap_att.layer.sublayer.kv_proj.", ".attn.kv_proj."),
        (".wrap_att.layer.sublayer.proj.", ".attn.proj."),
        (".wrap_att.layer.sublayer.rotary_embeddings.", ".attn.rotary_embeddings."),
        (".wrap_ff.layer.norm.", ".ff_norm."),
        (".wrap_ff.layer.sublayer.mlp.0.", ".feedforward.fc1."),
        (".wrap_ff.layer.sublayer.mlp.3.", ".feedforward.fc2."),
    )
    for old, new in replacements:
        if old in key:
            return key.replace(old, new)
    return key


def _upgrade_legacy_checkpoint_state_dict(state_dict):
    upgraded = {}
    for key, value in state_dict.items():
        new_key = _remap_legacy_checkpoint_key(key)
        if new_key in upgraded and new_key != key:
            continue
        upgraded[new_key] = value
    return upgraded


def _extract_zerograsp_model_state_dict(state_dict):
    upgraded = _upgrade_legacy_checkpoint_state_dict(state_dict)
    model_prefixed = {
        key[len("model.") :]: value
        for key, value in upgraded.items()
        if key.startswith("model.")
    }
    if model_prefixed:
        return model_prefixed
    return upgraded


class BaseTrainer(pl.LightningModule):
    GRASP_LOSS_KEYS = {
        "loss_validity",
        "loss_quality",
        "loss_tangent",
        "loss_approach",
        "loss_standoff",
        "loss_width",
        "loss_grasp_distill",
        "loss_grasp_topk_distill",
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.model = ZeroGrasp(config)
        self.apply_fine_tuning_freeze()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        upgraded_state_dict = _upgrade_legacy_checkpoint_state_dict(state_dict)
        return super().load_state_dict(
            upgraded_state_dict,
            strict=strict,
            assign=assign,
        )

    def load_checkpoint_weights(self, checkpoint_path):
        checkpoint = th.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            print(
                "Loaded checkpoint weights with missing/unexpected keys:",
                missing,
                unexpected,
            )

    def configure_distillation_teacher(self):
        if not getattr(self.config, "use_grasp_distillation", False):
            return
        checkpoint_path = getattr(self.config, "distill_checkpoint", None) or getattr(
            self.config, "checkpoint", None
        )
        if not checkpoint_path:
            raise ValueError(
                "use_grasp_distillation=true requires distill_checkpoint or checkpoint"
            )
        teacher = ZeroGrasp(self.config)
        checkpoint = th.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_zerograsp_model_state_dict(checkpoint.get("state_dict", checkpoint))
        missing, unexpected = teacher.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Teacher checkpoint load mismatch: missing={missing}, unexpected={unexpected}"
            )
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        self.teacher_model = teacher

    @staticmethod
    def set_module_trainable(module, trainable):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = trainable

    def apply_fine_tuning_freeze(self):
        if getattr(self.config, "train_grasp_head_only", False):
            for param in self.model.parameters():
                param.requires_grad = False
            self.set_module_trainable(self.model.predict, True)
            return

        if getattr(self.config, "freeze_image_encoder", False):
            self.set_module_trainable(self.model.backbone, False)

        if getattr(self.config, "freeze_reconstruction_branch", False):
            for module in (
                self.model.conv1,
                self.model.conv2,
                self.model.encoder_blks,
                self.model.downsample,
                self.model.upsample,
                self.model.decoder_blks,
                self.model.mae_encoder,
            ):
                self.set_module_trainable(module, False)

    def forward(self, batch):
        output = self.model(batch)

        occs = output["occs"]
        gt_occs = output["gt_occs"]
        loss_occ = 0
        stats_dict = {}
        for i, (occ, gt_occ) in enumerate(zip(occs, gt_occs)):
            depth = i + self.config.min_lod
            loss_occ += F.cross_entropy(occ, gt_occ)
            preds = occ.argmax(1).long()
            acc = binary_accuracy(preds, gt_occ.long())
            rec = binary_recall(preds, gt_occ.long())
            pre = binary_precision(preds, gt_occ.long())
            f1 = binary_f1_score(preds, gt_occ.long())
            stats_dict.update(
                {
                    f"acc_{depth}": acc,
                    f"rec_{depth}": rec,
                    f"pre_{depth}": pre,
                    f"f1_{depth}": f1,
                }
            )
        loss_dict = {"loss_occ": loss_occ}

        signal = output["signal"]
        gt_signal = output["gt_signal"]

        loss_nrm = th.mean(th.sum((gt_signal[:, :3] - signal[:, :3]) ** 2, dim=1))
        loss_sdf = th.mean((gt_signal[:, 3:4] - signal[:, 3:4]) ** 2)
        loss_dict["loss_nrm"] = loss_nrm
        loss_dict["loss_sdf"] = loss_sdf

        if self.config.predict_grasp:
            if "grasp_target_mask" in output:
                grasp_target_mask = output["grasp_target_mask"]
                stats_dict["grasp_mask_points"] = (
                    grasp_target_mask.sum(dim=1) > 0
                ).float().sum()
                stats_dict["grasp_mask_dims"] = grasp_target_mask.sum()
                grasp_loss_dict = self.compute_masked_grasp_loss(
                    signal[:, 4:14],
                    gt_signal[:, 4:14],
                    grasp_target_mask[:, :10],
                )
                loss_dict.update(grasp_loss_dict)
            else:
                grasp_target_mask = None
                valid = gt_signal[:, 4:5]
                loss_validity = th.mean((valid - signal[:, 4:5]) ** 2)
                loss_dict["loss_validity"] = 5.0 * loss_validity

                valid = valid.squeeze(-1) > 0.1
                if valid.sum() > 0:
                    loss_quality = th.mean(
                        (gt_signal[valid, 5:6] - signal[valid, 5:6]) ** 2
                    )
                    loss_tangent_1 = th.sum(
                        (gt_signal[valid, 6:9] - signal[valid, 6:9]) ** 2, dim=1
                    )
                    loss_tangent_2 = th.sum(
                        (-gt_signal[valid, 6:9] - signal[valid, 6:9]) ** 2, dim=1
                    )
                    loss_tangent = th.mean(th.minimum(loss_tangent_1, loss_tangent_2))
                    loss_approach = th.mean(
                        th.sum((gt_signal[valid, 9:12] - signal[valid, 9:12]) ** 2, dim=1)
                    )
                    loss_standoff = th.mean(
                        (gt_signal[valid, 12:13] - signal[valid, 12:13]) ** 2
                    )
                    loss_width = th.mean(
                        (gt_signal[valid, 13:14] - signal[valid, 13:14]) ** 2
                    )
                    loss_dict["loss_quality"] = 5.0 * loss_quality
                    loss_dict["loss_tangent"] = loss_tangent
                    loss_dict["loss_approach"] = loss_approach
                    loss_dict["loss_standoff"] = 2.5 * loss_standoff
                    loss_dict["loss_width"] = 2.5 * loss_width
                else:
                    loss_dict["loss_quality"] = 0.0
                    loss_dict["loss_tangent"] = 0.0
                    loss_dict["loss_approach"] = 0.0
                    loss_dict["loss_standoff"] = 0.0
                    loss_dict["loss_width"] = 0.0

            if getattr(self.config, "use_grasp_distillation", False):
                if not hasattr(self, "teacher_model"):
                    raise RuntimeError(
                        "Grasp distillation is enabled but teacher_model is not configured."
                    )
                with th.no_grad():
                    teacher_output = self.teacher_model(batch)
                    teacher_grasp = teacher_output["signal"][:, 4:14].detach()
                distill_loss = self.compute_grasp_distillation_loss(
                    signal[:, 4:14],
                    teacher_grasp,
                    grasp_target_mask[:, :10] if grasp_target_mask is not None else None,
                )
                loss_dict["loss_grasp_distill"] = (
                    float(getattr(self.config, "distill_weight", 0.0)) * distill_loss
                )
                stats_dict["loss_grasp_distill_unweighted"] = distill_loss
                if getattr(self.config, "use_grasp_topk_distillation", False):
                    batch_ids = output["octrees_out"].batch_id(
                        self.config.max_lod,
                        nempty=True,
                    )
                    topk_distill_loss = self.compute_grasp_topk_distillation_loss(
                        signal[:, 4:14],
                        teacher_grasp,
                        batch_ids,
                        grasp_target_mask[:, :10]
                        if grasp_target_mask is not None
                        else None,
                        topk=int(getattr(self.config, "distill_topk", 64)),
                        temperature=float(
                            getattr(self.config, "distill_temperature", 0.1)
                        ),
                    )
                    loss_dict["loss_grasp_topk_distill"] = (
                        float(getattr(self.config, "distill_topk_weight", 0.0))
                        * topk_distill_loss
                    )
                    stats_dict["loss_grasp_topk_distill_unweighted"] = (
                        topk_distill_loss
                    )

        if "pm" in output:
            global_step = self.trainer.global_step
            anneal_step = global_step % self.config.kl_loss_cycle_len
            anneal_start = 0
            anneal_end = (
                self.config.kl_loss_cycle_len // 2
            )  # optimize full weight for second half of cycle
            anneal_weight = (anneal_step - anneal_start) / (anneal_end - anneal_start)
            anneal_weight = 1.0 if anneal_weight > 1.0 else anneal_weight
            pm, pv, qm, qv = output["pm"], output["pv"], output["qm"], output["qv"]
            element_wise = 0.5 * (
                th.log(pv) - th.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
            )
            loss_kl_unweighted = element_wise.sum(-1).mean()
            # loss_dict['loss_kl'] = self.config.kl_weight * loss_kl_unweighted
            loss_dict["loss_kl"] = (
                anneal_weight * self.config.kl_weight * loss_kl_unweighted
            )
            stats_dict["kl_weight"] = anneal_weight
            stats_dict["loss_kl_unweighted"] = (
                self.config.kl_weight * loss_kl_unweighted
            )

        return loss_dict, stats_dict

    @staticmethod
    def masked_mean(value, mask):
        denom = mask.sum().clamp_min(1.0)
        return (value * mask).sum() / denom

    def compute_masked_grasp_loss(self, pred, target, target_mask):
        target_mask = target_mask.to(dtype=pred.dtype, device=pred.device).clamp(0.0, 1.0)
        target = target.to(dtype=pred.dtype, device=pred.device)
        zero = pred.sum() * 0.0

        loss_validity = self.masked_mean(
            (target[:, 0:1] - pred[:, 0:1]) ** 2,
            target_mask[:, 0:1],
        )
        loss_quality = self.masked_mean(
            (target[:, 1:2] - pred[:, 1:2]) ** 2,
            target_mask[:, 1:2],
        )

        tangent_mask = target_mask[:, 2:5].min(dim=1, keepdim=True).values
        if tangent_mask.sum() > 0:
            loss_tangent_1 = th.sum((target[:, 2:5] - pred[:, 2:5]) ** 2, dim=1, keepdim=True)
            loss_tangent_2 = th.sum((-target[:, 2:5] - pred[:, 2:5]) ** 2, dim=1, keepdim=True)
            loss_tangent = self.masked_mean(th.minimum(loss_tangent_1, loss_tangent_2), tangent_mask)
        else:
            loss_tangent = zero

        approach_mask = target_mask[:, 5:8].min(dim=1, keepdim=True).values
        if approach_mask.sum() > 0:
            loss_approach = self.masked_mean(
                th.sum((target[:, 5:8] - pred[:, 5:8]) ** 2, dim=1, keepdim=True),
                approach_mask,
            )
        else:
            loss_approach = zero

        loss_standoff = self.masked_mean(
            (target[:, 8:9] - pred[:, 8:9]) ** 2,
            target_mask[:, 8:9],
        )
        loss_width = self.masked_mean(
            (target[:, 9:10] - pred[:, 9:10]) ** 2,
            target_mask[:, 9:10],
        )

        return {
            "loss_validity": 5.0 * loss_validity,
            "loss_quality": 5.0 * loss_quality,
            "loss_tangent": loss_tangent,
            "loss_approach": loss_approach,
            "loss_standoff": 2.5 * loss_standoff,
            "loss_width": 2.5 * loss_width,
        }

    def compute_grasp_distillation_loss(self, pred, teacher_pred, target_mask=None):
        teacher_pred = teacher_pred.to(dtype=pred.dtype, device=pred.device)
        if teacher_pred.shape != pred.shape:
            raise ValueError(
                f"Teacher/student grasp shapes differ: {teacher_pred.shape} vs {pred.shape}"
            )
        if target_mask is None:
            mask = th.ones_like(pred)
        else:
            target_mask = target_mask.to(dtype=pred.dtype, device=pred.device).clamp(0.0, 1.0)
            mask = 1.0 - target_mask
        return self.masked_mean((pred - teacher_pred) ** 2, mask)

    def compute_grasp_topk_distillation_loss(
        self,
        pred,
        teacher_pred,
        batch_ids,
        target_mask=None,
        *,
        topk,
        temperature,
    ):
        if topk <= 0:
            return pred.sum() * 0.0
        temperature = max(float(temperature), 1e-4)
        batch_ids = batch_ids.to(device=pred.device).reshape(-1)
        if batch_ids.shape[0] != pred.shape[0]:
            raise ValueError(
                f"Batch-id/prediction lengths differ: {batch_ids.shape[0]} vs {pred.shape[0]}"
            )

        supervised_score = None
        if target_mask is not None:
            supervised_score = (
                target_mask[:, 0].to(device=pred.device, dtype=pred.dtype) > 0.0
            )

        losses = []
        for batch_id in th.unique(batch_ids):
            sample_mask = batch_ids == batch_id
            if supervised_score is not None:
                sample_mask = th.logical_and(sample_mask, ~supervised_score)
            sample_indices = th.nonzero(sample_mask, as_tuple=False).flatten()
            if sample_indices.numel() < 2:
                continue

            sample_teacher = teacher_pred[sample_indices, 0]
            sample_student = pred[sample_indices, 0]
            k = min(int(topk), int(sample_indices.numel()))
            teacher_top = th.topk(sample_teacher, k=k, sorted=False).indices
            student_top = th.topk(sample_student.detach(), k=k, sorted=False).indices
            union = th.unique(th.cat([teacher_top, student_top], dim=0))

            teacher_scores = sample_teacher[union]
            student_scores = sample_student[union]
            teacher_probs = F.softmax(teacher_scores / temperature, dim=0)
            student_log_probs = F.log_softmax(student_scores / temperature, dim=0)
            rank_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="sum",
            ) * (temperature**2)
            score_loss = F.mse_loss(
                sample_student[teacher_top],
                sample_teacher[teacher_top],
            )
            losses.append(rank_loss + score_loss)

        if not losses:
            return pred.sum() * 0.0
        return th.stack(losses).mean()

    def aggregate_loss(self, loss_dict):
        if not getattr(self.config, "grasp_head_only_loss", False):
            return sum(loss_dict.values())
        grasp_values = [
            value for name, value in loss_dict.items() if name in self.GRASP_LOSS_KEYS
        ]
        if not grasp_values:
            raise RuntimeError("grasp_head_only_loss=true but no grasp losses were produced")
        return sum(grasp_values)

    def training_step(self, batch, batch_idx):
        loss_dict, stats_dict = self(batch)
        loss = self.aggregate_loss(loss_dict)
        for name, value in loss_dict.items():
            self.log(
                f"train_{name}",
                value,
                on_step=True,
                prog_bar=True,
                sync_dist=True,
                rank_zero_only=True,
            )
        for name, value in stats_dict.items():
            self.log(
                f"train_{name}",
                value,
                on_step=True,
                prog_bar=True,
                sync_dist=True,
                rank_zero_only=True,
            )
        self.log(
            f"train_loss",
            loss,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss_dict, stats_dict = self(batch)
        loss = self.aggregate_loss(loss_dict)
        for name, value in loss_dict.items():
            self.log(
                f"valid_{name}",
                value,
                on_step=True,
                prog_bar=True,
                sync_dist=True,
                rank_zero_only=True,
            )
        for name, value in stats_dict.items():
            self.log(
                f"valid_{name}",
                value,
                on_step=True,
                prog_bar=True,
                sync_dist=True,
                rank_zero_only=True,
            )
        self.log(
            f"valid_loss",
            loss,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        return loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for optimizer")
        if self.config.optimizer == "Adam":
            optimizer = optim.Adam(trainable_params, lr=self.config.lr)
        elif self.config.optimizer == "AdamW":
            optimizer = optim.AdamW(
                trainable_params,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise Exception(f"{self.config.optimizer} is not supported!")

        scheduler = th.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.scheduler_step,
            gamma=self.config.scheduler_decay,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def train_dataloader(self):
        url = self.config.train_dataset_url
        if url.startswith("s3"):
            url = f"pipe:aws s3 cp {url} -"
        batch_size = self.config.batch_size
        num_workers = self.config.num_workers
        max_epochs = self.config.max_epochs
        dataset_size = self.config.train_dataset_size
        iter_per_epoch = dataset_size // (batch_size * self.trainer.num_devices)

        dataset = (
            wds.WebDataset(
                url,
                nodesplitter=wds.split_by_node,
                handler=wds.warn_and_continue,
                shardshuffle=True,
            )
            .decode(decode_depth, "pil", handler=wds.warn_and_continue)
            .map(
                make_sample_wrapper(
                    self.config, K=INTRINSICS_K[self.config.train_dataset_name]
                ),
                handler=wds.warn_and_continue,
            )
            .batched(batch_size)
        )

        dataloader = (
            wds.WebLoader(
                dataset,
                batch_size=None,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
            )
            .repeat(max_epochs)
            .with_epoch(iter_per_epoch)
            .with_length(iter_per_epoch)
        )

        return dataloader

    def val_dataloader(self):
        url = self.config.val_dataset_url
        if url.startswith("s3"):
            url = f"pipe:aws s3 cp {url} -"
        dataset_size = self.config.val_dataset_size

        dataset = (
            wds.WebDataset(
                url,
                nodesplitter=wds.split_by_node,
                handler=wds.warn_and_continue,
                shardshuffle=False,
            )
            .decode(decode_depth, "pil")
            .map(
                make_sample_wrapper(
                    self.config, K=INTRINSICS_K[self.config.val_dataset_name]
                ),
                handler=wds.warn_and_continue,
            )
            .batched(1)
        )

        dataloader = (
            wds.WebLoader(
                dataset, batch_size=None, shuffle=False, num_workers=0, pin_memory=False
            )
            .with_epoch(dataset_size // (self.trainer.num_devices))
            .with_length(dataset_size // (self.trainer.num_devices))
        )

        return dataloader

    def on_load_checkpoint(self, checkpoint):
        if self.config.fine_tuning:
            print("Reset the optimizer states")
            checkpoint["optimizer_states"] = []

    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            return
        for key in list(state_dict.keys()):
            if key.startswith("teacher_model."):
                del state_dict[key]


def main():
    config = parse_config()
    model = BaseTrainer(config)
    if config.fine_tuning and config.checkpoint:
        model.load_checkpoint_weights(config.checkpoint)
    model.configure_distillation_teacher()
    callbacks = []
    strategy = (
        "ddp_find_unused_parameters_true"
        if th.cuda.is_available() and th.cuda.device_count() > 1
        else "auto"
    )

    if config.mode == "training":
        # Store configurations in WandB
        checkpoint_path = os.path.join(
            config.default_root_dir, "checkpoints", config.project_name, config.run_name
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_path,
                save_top_k=-1,
                save_on_train_epoch_end=True,
                every_n_train_steps=config.checkpoint_every_n_steps,
            )
        )
        if config.logger == "wandb":
            logger = WandbLogger(
                project=config.project_name,
                entity=config.wandb_entity,
                name=config.run_name,
                log_model=True,
                save_dir=config.default_root_dir,
            )
        elif config.logger == "tensorboard":
            logger = TensorBoardLogger(
                save_dir=os.path.join(config.default_root_dir, "logs"),
                name=config.project_name,
                version=config.run_name,
            )
        else:
            logger = False
    else:
        config.batch_size = 1
        logger = TensorBoardLogger(
            save_dir=os.path.join(config.default_root_dir, "logs"),
            name=config.project_name,
            version=config.run_name or config.mode,
        )

    if logger:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        default_root_dir=config.default_root_dir,
        logger=logger,
        log_every_n_steps=config.log_every_n_steps,
        strategy=strategy,
        precision=32,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.5,
        callbacks=callbacks,
    )

    if trainer.global_rank == 0 and isinstance(logger, WandbLogger):
        logger.experiment.config.update(vars(config))
    fit_ckpt_path = None if config.fine_tuning else config.checkpoint
    trainer.fit(model=model, ckpt_path=fit_ckpt_path)


if __name__ == "__main__":
    main()
