import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor  # <- use threads

# -------------------------
# Module-level worker
# -------------------------
def _evaluate_val(args):
    cls, hparams, best_params, param, val, x, y, fast = args
    valid_args = set(hparams.keys())
    tmp_params = {**hparams, **best_params, param: val}
    safe_params = {k: v for k, v in tmp_params.items() if k in valid_args}

    tmp_model = cls(**safe_params)

    start_model_time = time.time()
    if fast:
        rng = np.random.default_rng(42)
        n = max(1, int(len(x) * 1.0))
        idx = rng.choice(len(x), size=n, replace=False)
        x_sub = [x[i] for i in idx]
        y_sub = [y[i] for i in idx]

        tmp_model.hparams["max_epochs"] = 50
        tmp_model.fit(x_sub, y_sub)
    else:
        tmp_model.fit(x, y)
    elapsed_model_time = time.time() - start_model_time

    loss = float(tmp_model._trainer.callback_metrics["val_loss"])
    return val, loss, elapsed_model_time

# -------------------------
# StepwiseHopt class
# -------------------------
class StepwiseHopt:
    def hopt(self, x, y, param_grid, fast=False, n_jobs=8, verbose=True):

        best_params = {}

        valid_args = set(self.hparams.keys())
        filtered_grid = {k: v for k, v in param_grid.items() if k in valid_args}

        total_steps = sum(len(v) for v in filtered_grid.values() if isinstance(v, (list, tuple)))
        current_step = 0
        start_time = time.time()

        for param, options in filtered_grid.items():
            if not isinstance(options, (list, tuple)):
                best_params[param] = options
                continue

            best_val = None
            best_loss = float('inf')

            if verbose:
                print(f"Optimizing hyperparameter: {param} ({len(options)} options)")

            args_list = [
                (self.__class__, self.hparams, best_params, param, val, x, y, fast)
                for val in options
            ]

            # -------------------------
            # Use threads for memory efficiency
            # -------------------------
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(_evaluate_val, args_list))

            # -------------------------
            # Collect results and update best
            # -------------------------
            for val, loss, model_time in results:
                current_step += 1
                progress_pct = (current_step / total_steps) * 100
                elapsed_min = model_time / 60.0  # show per-model time in minutes

                if verbose:
                    print(f"[{current_step}/{total_steps} | {progress_pct:3.1f}% | {elapsed_min:3.1f} min] "
                          f"Value: {str(val)}, val_loss: {loss:.4f}")

                if loss < best_loss:
                    best_loss = loss
                    best_val = val

            best_params[param] = best_val
            if verbose and best_val is not None:
                print(f"Best {param} = {str(best_val)}, val_loss = {best_loss:.4f}")

        self.hparams.update(best_params)
        total_time_min = (time.time() - start_time) / 60.0
        if verbose:
            print(f"Stepwise optimization completed in {total_time_min:.1f} min\n")

        return best_params
