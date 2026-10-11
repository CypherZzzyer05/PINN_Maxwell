import time

import torch


def train(model, adam_steps=1500, lbfgs_steps=400, samples_per_layer=64, seed=7):
    torch.manual_seed(seed)
    history = []
    started = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    samples = model.sample(samples_per_layer)

    def record(stage, step, loss):
        item = {"stage": stage, "step": step, "loss": float(loss.detach()),
                "neff": float(model.neff.detach()), "elapsed_s": time.perf_counter()-started}
        history.append(item)
        print(f"{stage} {step:5d}: loss={item['loss']:.3e} neff={item['neff']:.9f}", flush=True)

    for step in range(adam_steps):
        if step % 100 == 0:
            samples = model.sample(samples_per_layer)
        optimizer.zero_grad()
        loss = model.objective(samples)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == adam_steps-1:
            record("adam", step, loss)
    if lbfgs_steps:
        # Fixed collocation for line search. Twice Adam density.
        samples = model.sample(2*samples_per_layer, random=False)
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_steps,
                                      tolerance_grad=1e-11, tolerance_change=1e-13,
                                      history_size=50, line_search_fn="strong_wolfe")
        calls = 0

        def closure():
            nonlocal calls
            optimizer.zero_grad()
            loss = model.objective(samples)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite L-BFGS loss")
            loss.backward()
            calls += 1
            if calls % 100 == 0:
                record("lbfgs", calls, loss)
            return loss

        optimizer.step(closure)
        record("lbfgs_final", calls, model.objective(samples))
    terms = model.loss_terms(model.sample(513, random=False))
    return history, {key: float(value.detach()) for key, value in terms.items()}
