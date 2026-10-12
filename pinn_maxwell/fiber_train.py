import time

import torch


def train_fiber(model, adam_steps=2500, lbfgs_steps=1200, samples=96, seed=11):
    torch.manual_seed(seed)
    started = time.perf_counter()
    history = []

    def record(stage, step, loss):
        item = {"stage": stage, "step": step, "loss": float(loss.detach()),
                "neff": float(model.neff.detach()), "elapsed_s": time.perf_counter()-started}
        history.append(item)
        print(f"{stage} {step:5d}: loss={item['loss']:.3e} neff={item['neff']:.9f}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    points = model.sample(samples)
    for step in range(adam_steps):
        if step % 100 == 0:
            points = model.sample(samples)
        optimizer.zero_grad()
        loss = model.objective(points)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite fiber training loss")
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == adam_steps-1:
            record("adam", step, loss)
    if lbfgs_steps:
        points = model.sample(2*samples, random=False)
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_steps,
                                      tolerance_grad=1e-11, tolerance_change=1e-13,
                                      history_size=80, line_search_fn="strong_wolfe")
        calls = 0

        def closure():
            nonlocal calls
            optimizer.zero_grad()
            loss = model.objective(points)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite fiber L-BFGS loss")
            # A common scale preserves the physical objective and relative
            # weights, while avoiding premature absolute line-search tolerances
            # once the unscaled residual falls below 1e-6.
            scaled_loss = 1e5*loss
            scaled_loss.backward()
            calls += 1
            if calls % 200 == 0:
                record("lbfgs", calls, loss)
            return scaled_loss

        optimizer.step(closure)
        record("lbfgs_final", calls, model.objective(points))
    terms = model.loss_terms(model.sample(513, random=False))
    return history, {k: float(v.detach()) for k, v in terms.items()}
