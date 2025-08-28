import time

class StepwiseHopt:
    def hopt(self, x, y, param_grid, verbose=True):

        best_params = {}

        # use hparams keys as valid hyperparameters
        valid_args = set(self.hparams.keys())

        # filter param_grid to only include valid params
        filtered_grid = {k: v for k, v in param_grid.items() if k in valid_args}

        # Count only optimizable options for progress tracking
        total_steps = sum(len(v) for v in filtered_grid.values() if isinstance(v, (list, tuple)))
        current_step = 0
        start_time = time.time()

        for param, options in filtered_grid.items():
            # Skip fixed hyperparameters
            if not isinstance(options, (list, tuple)):
                best_params[param] = options
                continue

            best_val = None
            best_loss = float('inf')

            if verbose:
                print(f"Optimizing hyperparameter: {param} ({len(options)} options)")

            for val in options:
                current_params = {**self.hparams, **best_params, param: val}
                safe_params = {k: v for k, v in current_params.items() if k in valid_args}

                tmp_model = self.__class__(**safe_params)
                tmp_model.fit(x, y)
                loss = float(tmp_model._trainer.callback_metrics["val_loss"])

                current_step += 1
                progress_pct = (current_step / total_steps) * 100
                elapsed_min = (time.time() - start_time) / 60.0

                if verbose:
                    print(f"[{current_step}/{total_steps} | {progress_pct:3.1f}% | {elapsed_min:3.1f} min] "
                          f"Value: {str(val)}, val_loss: {loss:.4f}")

                if loss < best_loss:
                    best_loss = loss
                    best_val = val

            best_params[param] = best_val
            if verbose and best_val is not None:
                print(f"Best {param} = {str(best_val)}, val_loss = {best_loss:.4f}\n")

        # Update model hyperparameters
        self.hparams.update(best_params)
        total_time_min = (time.time() - start_time) / 60.0
        if verbose:
            print(f"Stepwise optimization completed in {total_time_min:.1f} min\n")

        return best_params
