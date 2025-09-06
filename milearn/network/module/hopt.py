import os
import time
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def get_optimal_torch_threads(n_jobs: int) -> int:
    total_cpus = os.cpu_count() or 1  # fallback to 1 if detection fails
    return max(1, total_cpus // n_jobs)

class StepwiseHopt:
    def _evaluate_model(self, cls, hparams, best_params, param, val, x, y, n_jobs):

        # limit torch threads for this trial
        torch.set_num_threads(get_optimal_torch_threads(n_jobs))

        valid_args = set(hparams.keys())
        tmp_params = {**hparams, **best_params, param: val}
        safe_params = {k: v for k, v in tmp_params.items() if k in valid_args}

        tmp_model = cls(**safe_params)

        start_model_time = time.time()
        tmp_model.fit(x, y)
        elapsed_model_time = time.time() - start_model_time

        epochs_trained = tmp_model._trainer.current_epoch + 1
        loss = float(tmp_model._trainer.callback_metrics["val_loss"])
        return val, loss, epochs_trained, elapsed_model_time

    def hopt(self, x, y, param_grid, verbose=True):

        # 1. Filter hparams
        valid_args = set(self.hparams.keys())
        filtered_grid = {k: v for k, v in param_grid.items() if k in valid_args}

        # 2. Start logging
        total_steps = sum(len(v) for v in filtered_grid.values() if isinstance(v, (list, tuple)))
        current_step = 0
        start_time = time.time()


        # 3. Start stepwise optimization
        best_params = {}
        for param, options in filtered_grid.items():
            # 3.1 Add fixed hparams (not list or tuple)
            if not isinstance(options, (list)): # TODO option can be tuple layers size
                best_params[param] = options
                continue

            best_val = None
            best_loss = float('inf')

            if verbose:
                print(f"Optimizing hyperparameter: {param} ({len(options)} options)")

            # 3.2 Prepare the list of models
            n_jobs = len(options)
            args_list = [
                (self.__class__, self.hparams, best_params, param, val, x, y, n_jobs)
                for val in options
            ]

            # 3.3 Start multi-thread hparams evaluation
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(lambda args: self._evaluate_model(*args), args_list))

            # 3.4 Collect and print the results
            for val, loss, epochs, model_time in results:
                current_step += 1
                progress_pct = (current_step / total_steps) * 100
                elapsed_min = model_time / 60.0  # show per-model time in minutes

                if verbose:
                    print(f"[{current_step}/{total_steps} | {progress_pct:4.1f}% | {elapsed_min:4.1f} min] "
                          f"Value: {str(val)}, Epochs: {epochs}, Loss: {loss:.4f}")

                if loss < best_loss:
                    best_loss = loss
                    best_val = val

            best_params[param] = best_val

            if verbose and best_val is not None:
                print(f"Best {param} = {str(best_val)}, val_loss = {best_loss:.4f}")

        # 4. Update with the found final best hparams
        self.hparams.update(best_params)

        # 5. Finish optimization
        total_time_min = (time.time() - start_time) / 60.0
        if verbose:
            print(f"Stepwise optimization completed in {total_time_min:.1f} min\n")

        return best_params
