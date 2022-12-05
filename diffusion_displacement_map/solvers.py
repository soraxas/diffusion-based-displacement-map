import abc
from typing import List

import torch.optim
from torch.optim.lr_scheduler import CyclicLR

from diffusion_displacement_map.critics import Critic


class Optimiser(abc.ABC):
    def __init__(self, lr: float):
        self.lr = lr
        self.iteration = 0
        self.scheduler = CyclicLR(
            self.optimizer,
            base_lr=self.lr,
            max_lr=self.lr * 2,
            step_size_up=5,
            cycle_momentum=False,
        )

    def update_lr(self, factor=None):
        print("warm")
        return
        if factor is not None:
            self.lr *= factor

        # The first forty iterations, we increase the learning rate slowly to full value.
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr * min(self.iteration / 40.0, 1.0) ** 2

    @abc.abstractmethod
    def step(self):
        ...


class SolverLBFGS(Optimiser):
    """Encapsulate the L-BFGS optimizer from PyTorch with a standard interface."""

    def __init__(self, objective, image, lr=1.0):
        self.objective = objective
        self.image = image
        self.retries = 0
        self.last_result = (float("+inf"), None)
        self.last_image = None

        self.lr = lr
        self.reset_optimizer()
        super().__init__(lr=lr)

    def reset_optimizer(self):
        print("==================")
        print("RESET optimiser")
        print("==================")
        self.optimizer = torch.optim.LBFGS(
            [self.image], lr=self.lr, max_iter=2, max_eval=4, history_size=10
        )
        self.iteration = 0

    def call_objective(self):
        """This function wraps the main LBFGS optimizer from PyTorch and uses simple
        hard-coded heuristics to determine its stability, and how best to manage it.

        Informally, it acts like a look-ahead optimizer that rolls back the step if
        there was a divergence in the optimization.
        """
        # Update the learning-rate dynamically, prepare optimizer.
        self.iteration += 1
        # self.update_lr()

        # Each iteration we reset the accumulated gradients.
        self.optimizer.zero_grad()

        # Prepare the image for evaluation, then run the objective.
        self.image.data.clamp_(0.0, 1.0)
        with torch.enable_grad():
            loss, scores = self.objective(self.image)
            self.next_result = loss, scores
            return loss

        # # Did the optimizer make progress as expected?
        # cur_result = self.image.grad.data.abs().mean()
        # if cur_result <= self.last_result[0] * 8.0:
        #     self.next_result = loss, scores
        #
        #     if cur_result < self.last_result[0] * 2.0:
        #         self.last_image = self.image.data.cpu().clone()
        #         self.last_result = (cur_result.item(), loss)
        #     return loss * 1.0
        #
        # # Look-ahead failed, so restore the image from the backup.
        # self.image.data[:] = self.last_image.to(self.image.device)
        # self.image.data[:] += torch.empty_like(self.image.data).normal_(std=1e-3)
        #
        # # There was a small regression: dampen the gradients and reduce step size.
        # if cur_result < self.last_result[0] * 24.0:
        #     self.image.grad.data.mul_(self.last_result[0] / cur_result)
        #     self.update_lr(factor=0.95)
        #     self.next_result = loss, scores
        #     return loss * 2.0
        #
        # self.update_lr(factor=0.8)
        # raise ValueError

    def step(self):
        """Perform one iteration of the optimization process.  This function will catch
        the optimizer diverging, and reset the optimizer's internal state.
        """
        while True:
            try:
                # This optimizer decides when and how to call the objective.
                self.optimizer.step(self.call_objective)
                self.scheduler.step()
                break
            except ValueError as e:
                print(e)
                # raise
                # To restart the optimization, we create a new instance from same image.
                self.reset_optimizer()
                self.retries += 1

        # Report progress once the first few retries are done.
        loss, score = self.next_result
        return loss, score


class SolverSGD(Optimiser):
    """Encapsulate the SGD or Adam optimizers from PyTorch with a standard interface."""

    def __init__(self, objective, image, opt_class="SGD", lr=1.0):
        opt_class = "Adam"
        self.objective = objective
        self.image = image
        self.retries = 0

        self.optimizer = getattr(torch.optim, opt_class)([image], lr=lr)
        super().__init__(lr=lr)

    def step(self):
        self.iteration += 1
        self.update_lr()

        # Each iteration we reset the accumulated gradients and compute the objective.
        self.optimizer.zero_grad()

        # Let the objective compute the loss and its gradients.
        self.image.data.clamp_(0.0, 1.0)
        assert self.image.requires_grad is True
        with torch.enable_grad():
            loss, scores = self.objective(self.image)

        assert self.image.grad is not None, "Objective did not produce image gradients."
        assert not torch.isnan(self.image.grad).any(), f"Gradient is NaN, loss {loss}."

        # Now compute the updates to the image according to the gradients.
        self.optimizer.step()
        self.scheduler.step()
        assert not torch.isnan(self.image).any(), f"Image is NaN for loss {loss}."

        return loss, scores


class Objective(abc.ABC):
    def __init__(
        self, encoder: torch.nn.Module, critics: List[Critic], alpha: float = None
    ):
        self.encoder = encoder
        self.critics = critics
        self.alpha_channel = alpha

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor):
        ...


class MultiCriticObjective(Objective):
    """An `Objective` that defines a problem to be solved by evaluating candidate
    solutions (i.e. images) and returning the computed error.

    This objective evaluates a list of critics to produce a final "loss" that's the sum
    of all the scores returned by the critics.  It's also responsible for computing the
    gradients.
    """

    def __call__(self, image):
        """Main evaluation function that's called by the solver.  Processes the image,
        computes the gradients, and returns the loss.
        """

        # with torch.autocast('cuda'):
        if 1:

            # Extract features from image.
            layers = [c.get_layers() for c in self.critics]
            feats = dict(self.encoder.extract(image, layers))

            scores = []

            # Apply all the critics one by one.
            # print(len(layers), len(self.critics))
            for critic in self.critics:
                critic.to(list(feats.values())[0])
                scores.append(sum(critic.evaluate(feats)))

            # Calculate the final loss and compute the gradients.
            loss = (sum(scores) / len(scores)).mean()

        loss.backward()

        if self.alpha_channel is not None:
            image.grad.data.mul_(self.alpha_channel)

        # from icecream import ic
        # image.grad.data[:, :, :image.shape[2] // 2, :image.shape[3] // 2] *= .0

        return loss.item(), scores


class SequentialCriticObjective(Objective):
    """An `Objective` that evaluates each of the critics one by one."""

    def __call__(self, image):
        # Apply all the critics one by one, keep track of results.
        scores = []
        for critic in self.critics:

            # Extract minimal necessary features from image.
            origin_feats = dict(
                self.encoder.extract(image, critic.get_layers(), as_checkpoints=True)
            )
            detach_feats = {
                k: f.detach().requires_grad_(True) for k, f in origin_feats.items()
            }

            # Ask the critic to evaluate the loss.
            total = 0.0
            for i, loss in enumerate(critic.evaluate(detach_feats)):
                assert not torch.isnan(loss), "Loss diverged to NaN."
                loss.backward()
                total += loss.item()
                del loss

            scores.append(total)

            # Backpropagate from those features.
            tensors, grads = [], []
            for original, optimized in zip(
                origin_feats.values(), detach_feats.values()
            ):
                if optimized.grad is not None:
                    tensors.append(original)
                    grads.append(optimized.grad)

            torch.autograd.backward(tensors, grads)

            del tensors
            del grads
            del origin_feats
            del detach_feats

        if self.alpha_channel is not None:
            image.grad.data.mul_(self.alpha_channel)

        return sum(scores) / len(scores), scores
