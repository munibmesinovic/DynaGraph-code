import numpy as np
import torch


class RunningImportance:
    def __init__(self, model, decay=0.9, rate=0.1):
        self.model = model
        self.decay, self.rate = decay, rate
        self._sum = torch.zeros_like(model.importance, dtype=torch.float64)
        self._n = 0
        self.steps = 0
        self.by_epoch, self.steps_by_epoch = [], []

    @torch.no_grad()
    def accumulate(self, G):
        if G.grad is None:
            raise RuntimeError("fused graph has no gradient")
        if tuple(G.grad.shape[1:]) != tuple(self.model.importance.shape):
            raise ValueError(f"fused graph shape {tuple(G.grad.shape)} does not match I {tuple(self.model.importance.shape)}")
        self._sum += G.grad.abs().sum(dim=0, dtype=torch.float64)
        self._n += G.grad.shape[0]

    @torch.no_grad()
    def step(self):
        I = self.model.importance
        I.copy_(self.decay * I.double() + self.rate * (self._sum / self._n))
        self._sum.zero_()
        self._n = 0
        self.steps += 1

    def end_epoch(self):
        self.by_epoch.append(self.model.importance.cpu().numpy().copy())
        self.steps_by_epoch.append(self.steps)

    def save(self, out_dir, selected_epoch, checkpoint_importance):
        by_epoch = np.stack(self.by_epoch)
        I = by_epoch[selected_epoch - 1]
        if not np.array_equal(I, checkpoint_importance.cpu().numpy()):
            raise RuntimeError("I of the selected epoch differs from the checkpoint")
        np.savez(out_dir / "importance.npz", I=I, node_scores=I.mean(axis=-1),
                 selected_epoch=selected_epoch, steps_at_selected=self.steps_by_epoch[selected_epoch - 1])
        np.savez(out_dir / "importance_epochs.npz", I_by_epoch=by_epoch,
                 epochs=np.arange(1, len(by_epoch) + 1), steps_by_epoch=np.array(self.steps_by_epoch))
        return I
