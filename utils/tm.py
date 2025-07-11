import time

from tqdm import tqdm


def eta(pbar: tqdm) -> float:
    """
    Estimate the remaining time for the progress bar.

    Returns:
        Estimated remaining time in hours.
    """
    if pbar.disable:
        return 0
    info = pbar.format_dict
    rate, total, n = info["rate"], info["total"], info["n"]
    return (total - n) / rate / 3600 if rate else 0


def eta_eval(epoch: int, cfg: dict, evaluation_hours: float) -> float:
    total_eval = cfg.get("epoch_stop") or cfg["epochs"] - max(
        cfg.get("warmup_epochs", -1), 0)
    return evaluation_hours * (total_eval - epoch)


class Timer:

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.elapsed = self.end - self.start

    def get_elapsed(self):
        return self.elapsed

    def get_elapsed_hours(self):
        return self.elapsed / 3600
