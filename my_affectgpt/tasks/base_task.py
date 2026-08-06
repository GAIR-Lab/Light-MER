"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import logging

import torch
import torch.distributed as dist
from my_affectgpt.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from my_affectgpt.common.logger import MetricLogger, SmoothedValue
from my_affectgpt.common.registry import registry
from my_affectgpt.datasets.data_utils import prepare_sample

# main process: model, dataset, training, evaluation, ...
class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()
        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls() # 'affectgpt.tasks.video_text_pretrain.VideoTextPretrainTask'

    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """
        
        datasets = dict()
        datasets_cfg = cfg.datasets_cfg
        model_cfg = cfg.model_cfg
        assert len(datasets_cfg) > 0, "At least one dataset has to be specified."

        for name in datasets_cfg:
            dataset_cfg = datasets_cfg[name]
            ############################ dataset_config Post-processing ############################
            assert dataset_cfg is not None
            if dataset_cfg.face_or_frame.startswith('multi'):
                assert model_cfg.multi_fusion_type in ['attention', 'qformer']
            builder = registry.get_builder_class(name)(dataset_cfg, model_cfg) # 找到这个dataset对应的builder
            ########################################################################################
            dataset = builder.build_datasets() # 每个builder有自己的 build_datasets 函数
            dataset['train'].name = name
            if 'sample_ratio' in dataset_cfg:
                dataset['train'].sample_ratio = dataset_cfg.sample_ratio
            datasets[name] = dataset
        return datasets

    # training: one iter
    def train_step(self, model, samples):
        outputs = model(samples)
        loss = outputs["loss"]
        if "ot_hidden_loss" in outputs:
            self._last_ce_loss = outputs["ce_loss"].item()
            self._last_ot_hidden = outputs["ot_hidden_loss"].item()
            self._last_kl_loss = outputs["kl_loss"].item()
            self._last_ot_weight = outputs.get("ot_weight", None)
        elif "kl_loss" in outputs:
            self._last_ce_loss = outputs["ce_loss"].item()
            self._last_kl_loss = outputs["kl_loss"].item()
            self._last_ot_hidden = None
            self._last_ot_weight = None
        else:
            self._last_ce_loss = None
            self._last_kl_loss = None
            self._last_ot_hidden = None
            self._last_ot_weight = None
        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    # one epoch contains iters_per_epoch iters (see trains.config)
    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        inner_epoch = epoch 
        iters_per_epoch = lr_scheduler.iters_per_epoch
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr",   SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("ce_loss", SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("ot_hid", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("kl_loss", SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("ot_w", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch) # 'Train: data epoch: [0]'
        
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break
            
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled) # move all samples-tensor into cuda
            global_step = (inner_epoch - 1) * iters_per_epoch + i  # epoch 从 1 开始
            samples.update( # add new key-value into map
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                    "global_step": global_step,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            # (amp, scaler) for amp training
            # Use bfloat16 for autocast: Qwen3 natively uses bfloat16 (max ~3.4e38),
            # float16 (max ~65504) causes immediate overflow → NaN/Inf
            amp_dtype = torch.bfloat16
            if torch.__version__.startswith('2.4.0'):
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    loss = self.train_step(model=model, samples=samples)
            elif torch.__version__.startswith('2.1.0'):
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    loss = self.train_step(model=model, samples=samples)
            elif torch.__version__.startswith('2.9'):
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    loss = self.train_step(model=model, samples=samples)
            else:
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    loss = self.train_step(model=model, samples=samples)

            # Check for NaN loss before backward
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"[Step {i}] NaN/Inf loss detected: {loss.item()}, skipping this batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # 梯度累积时必须缩放 loss，否则累积 N 次 backward 会使梯度放大 N 倍
            # 日志仍记录原始 loss（不缩放）
            loss_for_backward = loss / accum_grad_iters
            
            if use_amp:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    # For AMP training: unscale gradients before clipping
                    scaler.unscale_(optimizer)
                    
                    # Gradient clipping (max_norm=1.0 更常见于 LoRA+小头)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logging.warning(f"[Step {i}] NaN/Inf gradient norm: {grad_norm.item()}, resetting scaler")
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()  # Update scaler state
                        continue
                    
                    # Log gradient norm periodically
                    if i % 500 == 0:
                        logging.info(f"[Step {i}] Gradient norm: {grad_norm.item():.4f}")
                    
                    scaler.step(optimizer)
                    scaler.update()                     
                else:
                    # For FP32 training: clip directly (max_norm=1.0)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logging.warning(f"[Step {i}] NaN/Inf gradient norm: {grad_norm.item()}, skipping update")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    if i % 500 == 0:
                        logging.info(f"[Step {i}] Gradient norm: {grad_norm.item():.4f}")
                    
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            # Log OT/KL distillation metrics
            if hasattr(self, '_last_ce_loss') and self._last_ce_loss is not None:
                metric_logger.update(ce_loss=self._last_ce_loss)
            if hasattr(self, '_last_ot_hidden') and self._last_ot_hidden is not None:
                metric_logger.update(ot_hid=self._last_ot_hidden)
            if hasattr(self, '_last_kl_loss') and self._last_kl_loss is not None:
                metric_logger.update(kl_loss=self._last_kl_loss)
            if hasattr(self, '_last_ot_weight') and self._last_ot_weight is not None:
                metric_logger.update(ot_w=self._last_ot_weight)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
