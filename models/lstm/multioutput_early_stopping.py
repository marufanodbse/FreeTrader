from tensorflow.keras.callbacks import Callback
import numpy as np


class MultiOutputEarlyStopping(Callback):
    def __init__(
        self,
        monitors,
        patience=5,
        min_delta=0.0,
        mode="max",
        verbose=1,
        restore_best_weights=True
    ):
        super().__init__()
        self.monitors = (
            monitors  # e.g., ["val_output_1_accuracy", "val_output_2_accuracy"]
        )
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best = -np.Inf if mode == "max" else np.Inf
        self.best_weights = None
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = -np.Inf if self.mode == "max" else np.Inf
        self.best_weights = None
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        try:
            # current = np.mean([logs[k] for k in self.monitors])
            current = np.min([logs[k] for k in self.monitors])
        except KeyError as e:
            missing = [k for k in self.monitors if k not in logs]
            raise ValueError(f"Missing metrics in logs: {missing}")

        if (self.mode == "max" and current > self.best + self.min_delta) or (
            self.mode == "min" and current < self.best - self.min_delta
        ):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            if self.verbose:
                print(
                    f"Epoch {epoch + 1}: improvement detected. Best composite = {self.best:.5f}"
                )
        else:
            self.wait += 1
            if self.verbose:
                print(
                    f"Epoch {epoch + 1}: no improvement. Wait {self.wait}/{self.patience}"
                )
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                self.model.stop_training = True
                if self.verbose:
                    print(f"Early stopping triggered at epoch {self.stopped_epoch}")
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    if self.verbose:
                        print("Model weights restored to best epoch.")
