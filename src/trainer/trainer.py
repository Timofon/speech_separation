from pathlib import Path

import pandas as pd

from src.metrics.tracker import MetricTracker
from src.metrics.si_sdri import SISDRiMetric
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        outputs = {
            "s1_predicted": outputs['s1_predicted'],
            "s2_predicted": outputs['s2_predicted'],
        }
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_audio(**batch)
        else:
            # Log Stuff
            self.log_audio(**batch)
            self.log_predictions(**batch)
    
    def log_audio(self, s1_audio, s2_audio, mix_audio, **batch):
        self.writer.add_audio("first_audio", s1_audio[0], 16000)
        self.writer.add_audio("second_audio", s2_audio[0], 16000)

        self.writer.add_audio("mix_audio", mix_audio[0], 16000)

    def log_predictions(
        self, s1_audio, s2_audio, s1_predicted, s2_predicted, mix_audio, **batch
    ):
        self.writer.add_audio("first_audio", s1_audio[0], 16000)
        self.writer.add_audio("first_predicted", s1_predicted[0], 16000)
        
        self.writer.add_audio("second_audio", s2_audio[0], 16000)
        self.writer.add_audio("second_predicted", s2_predicted[0], 16000)
        
        self.writer.add_audio("mix_audio", mix_audio[0], 16000)

        self.writer.add_scalar("SI-SDRi", SISDRiMetric(s1_audio, s2_audio, s1_predicted, s2_predicted, mix_audio))
