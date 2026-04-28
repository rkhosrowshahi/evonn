"""Parameter sharing strategies for low-dimensional evolutionary search."""

from __future__ import annotations

import argparse
import os
import math
import numpy as np
from typing import Any

import torch
import torch.nn as nn


def _z_to_torch_cpu(z: Any) -> torch.Tensor:
    """Flatten ``z`` to 1D float64 on CPU."""
    if isinstance(z, torch.Tensor):
        z = z.detach()
    return torch.as_tensor(z, dtype=torch.float64, device="cpu").reshape(-1)


def _to_device_float32(x: Any, device: Any) -> torch.Tensor:
    """Cast ``x`` to a float32 tensor on ``device``."""
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def adapter_checkpoint_dict(adapter: "ParameterAdapterBase") -> dict[str, Any]:
    """Snapshot adapter fields for :func:`torch.save` (excludes ``model``; tensors to CPU)."""
    st: dict[str, Any] = {}
    for k, v in adapter.__dict__.items():
        if k == "model":
            continue
        if isinstance(v, torch.Tensor):
            st[k] = v.detach().cpu().clone()
        elif isinstance(v, np.ndarray):
            st[k] = v.copy()
        else:
            st[k] = v
    return {"class": adapter.__class__.__name__, "state": st}


def _tensor_placed_for_load(name: str, t: torch.Tensor, model_device: torch.device) -> torch.Tensor:
    """Map checkpoint tensors (saved on CPU) back to devices expected by :meth:`apply`."""
    t = t.detach().clone()
    n = name.lower()
    if name == "z0" or n == "projections" or ("projection" in n and "base" not in n):
        return t.cpu()
    if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool, torch.uint8):
        return t.to(model_device)
    return t.to(model_device)


def load_adapter_checkpoint_state(
    adapter: "ParameterAdapterBase",
    payload: dict[str, Any],
    *,
    strict: bool = True,
    model_device: torch.device | str | None = None,
) -> None:
    """Restore :attr:`__dict__` fields saved with :func:`adapter_checkpoint_dict`.

    The adapter must already wrap the target ``model`` (``get_adapter(...)``) with compatible
    architecture; call :meth:`torch.nn.Module.load_state_dict` on the model first.

    Parameters
    ----------
    payload
        Either ``{ "class", "state" }`` as returned by :func:`adapter_checkpoint_dict`, or
        a raw ``state`` mapping (``class`` is then not checked).
    model_device
        Device for non-CPU tensors (e.g. ``base_params``, ``assignment``). Defaults to the
        first parameter of ``adapter.model``.
    """
    if "state" in payload:
        state = payload["state"]
        if strict and "class" in payload and payload["class"] != adapter.__class__.__name__:
            raise ValueError(
                f"adapter class mismatch: checkpoint has {payload['class']!r}, "
                f"got {adapter.__class__.__name__!r}"
            )
    else:
        state = payload

    if model_device is None:
        model_device = next(adapter.model.parameters()).device
    else:
        model_device = torch.device(model_device) if not isinstance(model_device, torch.device) else model_device

    for k, v in state.items():
        if k == "model":
            continue
        if isinstance(v, torch.Tensor):
            v = _tensor_placed_for_load(k, v, model_device)
        elif isinstance(v, np.ndarray):
            v = v.copy()
        setattr(adapter, k, v)


def load_evo_checkpoint(
    path: str | os.PathLike,
    *,
    model: nn.Module,
    adapter: "ParameterAdapterBase",
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load ``model_state_dict`` and ``adapter_state`` from an ``evo_trainer`` / ``evo_2phase_trainer`` file.

    Example::

        model = get_model(...)
        adapter = get_adapter("full", model=model, args=args)
        ckpt = load_evo_checkpoint("logs/run/checkpoints/best.pt", model=model, adapter=adapter)
    """
    p = os.fspath(path)
    # PyTorch 2.4+ supports weights_only; old checkpoints need full unpickle.
    try:
        ckpt = torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover
        ckpt = torch.load(p, map_location=map_location)
    if "model_state_dict" not in ckpt or "adapter_state" not in ckpt:
        raise KeyError("checkpoint must contain 'model_state_dict' and 'adapter_state'")
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    dev = next(model.parameters()).device
    load_adapter_checkpoint_state(adapter, ckpt["adapter_state"], strict=strict, model_device=dev)
    return ckpt


class ParameterAdapterBase:
    """Every adapter implements the same three steps, driven by :meth:`apply`.

    **Step 1 — latent → full flat: ``z → x``**
        :meth:`decode` maps the latent vector ``z`` to a full-length flat vector.
        :meth:`scale` then multiplies by ``alpha``: ``x = scale(decode(z), alpha)``.

    **Step 2 — compute network weights: ``(theta_0, x) → theta``**
        :meth:`compute_theta` determines whether ``x`` is a delta or a full vector:

        * **Residual** (default): ``theta = theta_0 + x``. ``x`` encodes the update;
          ``alpha`` inside :meth:`scale` already handles ``delta = x_raw * alpha``.
        * **Direct** (:class:`FullSpace`, :class:`GlobalUniformBinningDirectly`):
          ``theta = x``. Override :meth:`compute_theta` to ignore ``theta_0``.

    **Step 3 — load: ``theta → model``**
        :meth:`load_theta` writes the flat vector via
        :func:`torch.nn.utils.vector_to_parameters`.
    """

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent → full flat: map compressed ``z`` to a full-length vector (step 1a).

        Must be implemented by each adapter. Input ``z`` is a 1D float64 CPU tensor.
        """
        raise NotImplementedError

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Apply ``alpha`` to the decoded vector ``x`` (step 1b).

        Must be implemented by each adapter. Returns the scaled flat tensor.
        """
        raise NotImplementedError

    def compute_theta(self, theta_0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Residual: ``theta = theta_0 + x`` (step 2 default)."""
        return theta_0 + x

    def load_theta(self, theta: torch.Tensor) -> None:
        """Write flat weight vector into ``self.model`` (step 3)."""
        nn.utils.vector_to_parameters(theta, self.model.parameters())

    def apply(
        self,
        theta_0: torch.Tensor,
        z: Any,
        *,
        alpha: float,
        device: torch.device,
    ) -> None:
        """Run all three steps: ``scale(decode(z)) → compute_theta → load_theta``."""
        z_t = _z_to_torch_cpu(z)
        x = self.scale(self.decode(z_t), alpha=alpha)
        x = _to_device_float32(x, device)
        x = x.reshape_as(theta_0)
        theta = self.compute_theta(theta_0, x)
        self.load_theta(theta)


class FullSpace(ParameterAdapterBase):
    """Evolve the full flat weight vector: ``z in R^N`` is ``theta`` (no residual on ``theta_0``).

    :meth:`apply` sets ``theta = scale(decode(z))``; ``theta_0`` is ignored.
    ``alpha`` scales the whole vector in :meth:`scale`.

    ``num_dims`` equals the total parameter count. ``z0`` is the flat snapshot at
    construction (for any code that reads a reference point); optimizers still use
    ``evo_lb`` / ``evo_ub`` on coordinates of ``z``.
    """

    def __init__(self, model, device="cuda", seed=42):
        self.model = model
        self.device = device
        self._seed = int(seed)
        flat = nn.utils.parameters_to_vector(model.parameters())
        self.N = int(flat.numel())
        self.num_dims = self.N
        self.base_params = flat.detach().clone()
        self.z0 = self.base_params.cpu().to(torch.float64).reshape(-1)
        print(f"FullSpace: num_dims={self.num_dims} (full-parameter search)", flush=True)

    def compute_theta(self, theta_0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: Any) -> torch.Tensor:
        return _z_to_torch_cpu(z)

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        x = _to_device_float32(x, self.device)
        return float(alpha) * x

    def init_pop(self, type: str, pop_size: int, sigma: float = 0.1,
                 lb: float = -3.0, ub: float = 3.0):
        num_dims = int(self.z0.numel())
        pop = np.empty((pop_size, num_dims), dtype=np.float64)
        if type == "uniform":
            if pop_size:
                pop[:] = np.random.uniform(lb, ub, size=(pop_size, num_dims))
        elif type == "normal":
            if pop_size:
                pop[:] = np.random.normal(0.0, sigma, size=(pop_size, num_dims))
        else:
            raise ValueError(f"FullSpace.init_pop: unknown type {type!r}; choices: uniform, normal")
        return pop


class RandomProjection(ParameterAdapterBase):
    """Random projection from k-dimensional latent space to full parameter space.

    The latent vector has length ``k + 1``: the first ``k`` components map through
    a fixed random matrix ``P``; the last component is a scalar **s** that scales
    the projected delta: ``delta = s * (P @ z)``.  Initial ``s`` is 1 so that with
    ``z = 0`` the delta is still zero (``P @ 0 = 0``).
    """

    def __init__(self, model, k, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = len(self.base_params)
        self.k = k
        self.device = device

        self._rng = torch.Generator()
        self._rng.manual_seed(int(seed))
        torch.manual_seed(int(seed))

        self.projections = (
            torch.randn(self.N, self.k, generator=self._rng, dtype=torch.float64)
            / math.sqrt(self.k)
        )
        self.num_dims = self.k + 1
        self.z0 = self._init_z0()

    def _init_z0(self):
        z0 = torch.zeros(self.num_dims, dtype=torch.float64, device=torch.device("cpu"))
        z0[-1] = 1.0
        return z0

    def decode(self, z):
        """Map latent vector z to full parameter space via random projection."""
        z = _z_to_torch_cpu(z)
        z_lat = z[: self.k]
        s = float(z[self.k].item())
        x = self.projections @ z_lat
        return s * x

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Apply ``alpha`` scaling to the decoded flat vector."""
        return _to_device_float32(x, self.device) * alpha




class LayerwiseRandomProjection(ParameterAdapterBase):
    """Layer-wise random projection with direct bias evolution.

    Each non-bias parameter tensor gets its own projection matrix
    ``P_l in R^{n_l x k}``, so latent dimensionality is::

        num_dims = k * L + sum(bias_sizes)

    where ``L`` is the number of projected (non-bias) parameter tensors.
    Bias tensors are appended directly to ``z`` (no projection).
    """

    def __init__(self, model, k, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = len(self.base_params)
        self.k = k
        self.device = device

        self._rng = torch.Generator()
        self._rng.manual_seed(int(seed))
        torch.manual_seed(int(seed))

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        self.projections = []
        total_dims = 0
        n_projected_layers = 0
        n_bias_dims = 0

        for name, param in self.model.named_parameters():
            shape = param.shape
            size = int(math.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)

            if name.endswith(".bias"):
                dims = size
                self.layer_info.append(
                    {
                        "type": "direct",
                        "name": name,
                        "n": size,
                        "dims": dims,
                        "offset": total_dims,
                        "size": size,
                    }
                )
                total_dims += dims
                n_bias_dims += dims
            else:
                P = (
                    torch.randn(size, self.k, generator=self._rng, dtype=torch.float64)
                    / math.sqrt(self.k)
                )
                proj_idx = len(self.projections)
                self.projections.append(P)
                dims = self.k
                self.layer_info.append(
                    {
                        "type": "proj",
                        "name": name,
                        "dims": dims,
                        "offset": total_dims,
                        "size": size,
                        "proj_idx": proj_idx,
                        "k": self.k,
                    }
                )
                total_dims += dims
                n_projected_layers += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"LayerwiseRandomProj: L={n_projected_layers}, k={self.k}, "
            f"bias_dims={n_bias_dims} | z dim = {self.num_dims} | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        return torch.zeros(self.num_dims, dtype=torch.float64, device=torch.device("cpu"))

    def decode(self, z):
        """Expand layer-wise latent vector to full parameter-space delta."""
        z = _z_to_torch_cpu(z)

        reconstructed = []
        for info in self.layer_info:
            offset = info["offset"]
            dims = info["dims"]
            z_layer = z[offset : offset + dims]

            if info["type"] == "proj":
                P = self.projections[info["proj_idx"]]
                reconstructed.append(P @ z_layer)
            else:
                reconstructed.append(z_layer.clone())

        return torch.cat(reconstructed)

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Apply ``alpha`` scaling to the decoded flat vector."""
        return _to_device_float32(x, self.device) * alpha




class LayerwiseRandomBlocking(ParameterAdapterBase):
    """Layer-wise random blocking: k latent scalars per weight tensor.

    Each non-bias parameter tensor (flattened to length ``n_l``) is partitioned
    into ``k`` blocks by randomly assigning every weight index to one of ``k``
    subspace indices via a random assignment array ``assignment`` of length ``n_l``::

        delta[i] = z_l[assignment[i]]     for i in 0 … n_l-1

    All weights assigned to block ``j`` share the value ``z_l[j]``.

    If ``n_l < k``, that tensor is evolved **directly** (like bias): ``z_l`` has
    length ``n_l`` and ``delta = z_l``.

    Biases are always direct. Total latent dimension is
    ``sum_l min(k, n_l) + sum(bias_sizes)`` over weight tensors ``l`` and biases.
    """

    def __init__(self, model, k, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = len(self.base_params)
        self.k = k
        self.device = device

        self._rng = torch.Generator()
        self._rng.manual_seed(int(seed))
        torch.manual_seed(int(seed))

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        total_dims = 0
        n_blocking_layers = 0
        n_bias_dims = 0
        n_direct_weight_dims = 0

        for name, param in self.model.named_parameters():
            shape = param.shape
            size = int(math.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)

            if name.endswith(".bias"):
                dims = size
                self.layer_info.append(
                    {
                        "type": "direct",
                        "name": name,
                        "n": size,
                        "dims": dims,
                        "offset": total_dims,
                        "size": size,
                    }
                )
                total_dims += dims
                n_bias_dims += dims
            else:
                if self.k > size:
                    self.layer_info.append(
                        {
                            "type": "direct",
                            "name": name,
                            "n": size,
                            "dims": size,
                            "offset": total_dims,
                            "size": size,
                        }
                    )
                    total_dims += size
                    n_direct_weight_dims += size
                else:
                    n_tiles = int(math.ceil(size / self.k))
                    base = torch.arange(self.k, dtype=torch.int64).repeat(n_tiles)[:size]
                    perm = torch.randperm(size, generator=self._rng)
                    assignment = base[perm]
                    self.layer_info.append(
                        {
                            "type": "rand_blocking",
                            "name": name,
                            "dims": self.k,
                            "offset": total_dims,
                            "size": size,
                            "assignment": assignment,
                        }
                    )
                    total_dims += self.k
                    n_blocking_layers += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"LayerwiseRandomBlocking: blocking_layers={n_blocking_layers}, k={self.k}, "
            f"bias_dims={n_bias_dims}, direct_weight_dims={n_direct_weight_dims} | "
            f"z dim = {self.num_dims} | model params = {self.N}"
        )

    def _init_z0(self):
        return torch.zeros(self.num_dims, dtype=torch.float64, device=torch.device("cpu"))

    def decode(self, z):
        """Expand latent vector to full parameter-space delta via random index ties."""
        z = _z_to_torch_cpu(z)

        reconstructed = []
        for info in self.layer_info:
            offset = info["offset"]
            dims = info["dims"]
            z_layer = z[offset : offset + dims]

            if info["type"] == "rand_blocking":
                reconstructed.append(z_layer[info["assignment"]])
            else:
                reconstructed.append(z_layer.clone())

        return torch.cat(reconstructed)

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Apply ``alpha`` scaling to the decoded flat vector."""
        return _to_device_float32(x, self.device) * alpha




class GlobalUniformBinningWithDelta(ParameterAdapterBase):
    """Global **value-based** uniform binning over **all** model parameters (mixed).

    At construction, the flat parameter vector ``v`` (weights and biases, same order
    as :func:`torch.nn.utils.parameters_to_vector`) defines a 1D axis. Let
    ``delta = max(v) - min(v)``. With ``num_bins`` equal-width intervals covering
    ``[min(v), max(v)]``, each scalar is assigned the bin index of the interval that
    contains its value. **Empty bins are removed**; latent dimension ``num_dims`` is
    the count of bins that contain at least one parameter (so ``bins_used <= num_bins``;
    equality means every requested bin had at least one weight, which is common when
    the parameter count is large relative to ``num_bins``). Every parameter index
    shares the latent coordinate of its (compressed) bin — there is no separate
    direct-evolution path for biases or small tensors.

    If all values are equal (``delta == 0``), every entry maps to a single bin.
    """

    def __init__(self, model, num_bins, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = int(len(self.base_params))
        self.num_bins = int(num_bins)
        if self.num_bins < 1:
            raise ValueError("num_bins must be >= 1")
        self.device = device
        self._seed = int(seed)

        self.param_shapes = []
        self.param_sizes = []
        for _, param in self.model.named_parameters():
            shape = param.shape
            size = int(math.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)

        vals = self.base_params.detach().cpu().to(torch.float64).reshape(-1)
        vmin = float(vals.min().item())
        vmax = float(vals.max().item())
        span = vmax - vmin
        B = self.num_bins

        if self.N == 0:
            self.assignment = torch.zeros(0, dtype=torch.int64)
            self.num_dims = 0
            self._value_min = vmin
            self._value_max = vmax
            self._bins_requested = B
            self._bins_used = 0
        elif not math.isfinite(span) or span <= 0.0:
            bin_raw = torch.zeros(self.N, dtype=torch.int64)
            self._build_compressed_assignment(bin_raw, B, vmin, vmax)
        else:
            # Bin k covers [vmin + k * w, vmin + (k+1) * w), last bin closed on the right at vmax.
            w = span / float(B)
            t = (vals - vmin) / w
            bin_raw = torch.floor(t).to(torch.int64)
            bin_raw.clamp_(0, B - 1)
            # Ensure vmax lands in the last bin (floor can be B-1 already; guard float edge)
            bin_raw = torch.where(vals >= vmax, torch.tensor(B - 1, dtype=torch.int64), bin_raw)
            self._build_compressed_assignment(bin_raw, B, vmin, vmax)

        self.z0 = self._init_z0()

        dropped = int(self._bins_requested) - int(self._bins_used)
        drop_msg = f" empty_bins_dropped={dropped}" if dropped > 0 else " (no empty bins)"
        print(
            f"GlobalUniformBinning: global value bins | params={self.N} | "
            f"value_range=[{self._value_min:.6g},{self._value_max:.6g}] | "
            f"bins_requested={self._bins_requested} bins_used={self._bins_used}{drop_msg} | "
            f"z_dim={self.num_dims}",
            flush=True,
        )

    def _build_compressed_assignment(
        self,
        bin_raw: torch.Tensor,
        B: int,
        vmin: float,
        vmax: float,
    ) -> None:
        """Set ``assignment`` (length N, entries in 0..K-1) and ``num_dims`` = K."""
        counts = torch.bincount(bin_raw, minlength=B)
        used = torch.nonzero(counts > 0, as_tuple=False).reshape(-1)
        K = int(used.numel())
        remap = torch.full((B,), -1, dtype=torch.int64)
        if K > 0:
            remap[used] = torch.arange(K, dtype=torch.int64)
        self.assignment = remap[bin_raw]
        self.num_dims = K
        self._value_min = vmin
        self._value_max = vmax
        self._bins_requested = B
        self._bins_used = K

    def _init_z0(self):
        return torch.zeros(self.num_dims, dtype=torch.float64, device=torch.device("cpu"))

    def decode(self, z):
        """Decode ``z`` (one scalar per non-empty value bin) to a full-length flat vector."""
        z = _z_to_torch_cpu(z)
        if self.N == 0:
            return z
        return z[self.assignment]

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return _to_device_float32(x, self.device) * alpha


class GlobalUniformBinningDirectly(ParameterAdapterBase):
    """Global **value-based** uniform binning over **all** model parameters (mixed).

    At construction, the flat parameter vector ``v`` (weights and biases, same order
    as :func:`torch.nn.utils.parameters_to_vector`) defines a 1D axis. Let
    ``delta = max(v) - min(v)``. With ``num_bins`` equal-width intervals covering
    ``[min(v), max(v)]``, each scalar is assigned the bin index of the interval that
    contains its value. **Empty bins are removed**; latent dimension ``num_dims`` is
    the count of bins that contain at least one parameter (so ``bins_used <= num_bins``;
    equality means every requested bin had at least one weight, which is common when
    the parameter count is large relative to ``num_bins``). Every parameter index
    shares the latent coordinate of its (compressed) bin — there is no separate
    direct-evolution path for biases or small tensors.

    If all values are equal (``delta == 0``), every entry maps to a single bin.

    **Direct semantics.** :meth:`apply` sets ``theta = scale(decode(z))``; the
    trainer snapshot ``theta_0`` is ignored (see :meth:`compute_theta`). Here
    :meth:`scale` is ``alpha * decode(z)``, i.e. one scalar per (compressed) value
    bin shared by all parameters in that bin. The tensor :attr:`z0` stores per-bin
    **mean weight values** at construction; with ``alpha = 1``, ``forward(z0)``
    assigns each index the mean of its value bin, which seeds phase-2 DE on this
    subspace (not bit-identical to the pre-bin snapshot unless all weights in a bin
    agree).
    """

    def __init__(self, model, num_bins, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = int(len(self.base_params))
        self.num_bins = int(num_bins)
        if self.num_bins < 1:
            raise ValueError("num_bins must be >= 1")
        self.device = device
        self._seed = int(seed)

        self.param_shapes = []
        self.param_sizes = []
        for _, param in self.model.named_parameters():
            shape = param.shape
            size = int(math.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)

        vals = self.base_params.detach().cpu().to(torch.float64).reshape(-1)
        vmin = float(vals.min().item())
        vmax = float(vals.max().item())
        span = vmax - vmin
        B = self.num_bins

        if self.N == 0:
            self.assignment = torch.zeros(0, dtype=torch.int64)
            self.num_dims = 0
            self._value_min = vmin
            self._value_max = vmax
            self._bins_requested = B
            self._bins_used = 0
        elif not math.isfinite(span) or span <= 0.0:
            bin_raw = torch.zeros(self.N, dtype=torch.int64)
            self._build_compressed_assignment(bin_raw, B, vmin, vmax)
        else:
            # Bin k covers [vmin + k * w, vmin + (k+1) * w), last bin closed on the right at vmax.
            w = span / float(B)
            t = (vals - vmin) / w
            bin_raw = torch.floor(t).to(torch.int64)
            bin_raw.clamp_(0, B - 1)
            # Ensure vmax lands in the last bin (floor can be B-1 already; guard float edge)
            bin_raw = torch.where(vals >= vmax, torch.tensor(B - 1, dtype=torch.int64), bin_raw)
            self._build_compressed_assignment(bin_raw, B, vmin, vmax)

        if self.N == 0 or self.num_dims == 0:
            self.z0 = torch.zeros(self.num_dims, dtype=torch.float64, device=torch.device("cpu"))
            self.bin_lb = np.zeros(self.num_dims, dtype=np.float64)
            self.bin_ub = np.zeros(self.num_dims, dtype=np.float64)
        else:
            K = self.num_dims
            bp = self.base_params.detach().to(torch.float64)
            assn = self.assignment.to(self.base_params.device)
            sums = torch.zeros(K, dtype=torch.float64, device=self.base_params.device)
            counts = torch.zeros(K, dtype=torch.float64, device=self.base_params.device)
            sums.index_add_(0, assn, bp)
            counts.index_add_(0, assn, torch.ones_like(bp, dtype=torch.float64))
            self.z0 = (sums / counts).cpu()
            # Per-bin parameter value boundaries for population initialisation.
            bp_np = bp.cpu().numpy()
            assn_np = assn.cpu().numpy()
            self.bin_lb = np.empty(K, dtype=np.float64)
            self.bin_ub = np.empty(K, dtype=np.float64)
            for k in range(K):
                vals_k = bp_np[assn_np == k]
                self.bin_lb[k] = float(vals_k.min())
                self.bin_ub[k] = float(vals_k.max())

        dropped = int(self._bins_requested) - int(self._bins_used)
        drop_msg = f" empty_bins_dropped={dropped}" if dropped > 0 else " (no empty bins)"
        print(
            f"GlobalUniformBinning: global value bins | params={self.N} | "
            f"value_range=[{self._value_min:.6g},{self._value_max:.6g}] | "
            f"bins_requested={self._bins_requested} bins_used={self._bins_used}{drop_msg} | "
            f"z_dim={self.num_dims}",
            flush=True,
        )

    def compute_theta(self, theta_0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x

    def _build_compressed_assignment(
        self,
        bin_raw: torch.Tensor,
        B: int,
        vmin: float,
        vmax: float,
    ) -> None:
        """Set ``assignment`` (length N, entries in 0..K-1) and ``num_dims`` = K."""
        counts = torch.bincount(bin_raw, minlength=B)
        used = torch.nonzero(counts > 0, as_tuple=False).reshape(-1)
        K = int(used.numel())
        remap = torch.full((B,), -1, dtype=torch.int64)
        if K > 0:
            remap[used] = torch.arange(K, dtype=torch.int64)
        self.assignment = remap[bin_raw]
        self.num_dims = K
        self._value_min = vmin
        self._value_max = vmax
        self._bins_requested = B
        self._bins_used = K

    def _init_z0(self):
        return torch.zeros(self.num_dims, dtype=torch.float64, device=torch.device("cpu"))

    def decode(self, z):
        """Decode ``z`` (one scalar per non-empty value bin) to a full-length flat vector."""
        z = _z_to_torch_cpu(z)
        if self.N == 0:
            return z
        return z[self.assignment]

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return _to_device_float32(x, self.device) * alpha



    def init_pop(
        self,
        type: str,
        pop_size: int,
        sigma: float = 0.1,
        lb: float | None = None,
        ub: float | None = None,
    ):
        # lb/ub mirror :meth:`FullSpace.init_pop` and the phase-2 trainer call site.
        # Binned ``best+...`` modes use per-bin or best-vector ranges, not these bounds.

        if type == "best+normal_on_centroid":
            # Direct: theta = forward(z). First individual z = z0 (per-bin means) so
            # initial weights sit in the bin-tied subspace at the value-bin centroid.
            best = np.asarray(self.z0.cpu().numpy(), dtype=np.float64).reshape(-1)
            centroid = float(np.mean(best)) if best.size else 0.0
            pop = np.empty((pop_size, best.size), dtype=np.float64)
            if pop_size:
                pop[0] = best
            if pop_size > 1:
                pop[1:] = np.random.normal(centroid, sigma, size=(pop_size - 1, best.size))
            return pop
        elif type == "best+uniform_on_best_lb_ub":
            best = np.asarray(self.z0.cpu().numpy(), dtype=np.float64).reshape(-1)
            zmin = float(best.min()) if best.size else 0.0
            zmax = float(best.max()) if best.size else 0.0
            if zmin > zmax:
                zmin, zmax = zmax, zmin
            pop = np.empty((pop_size, best.size), dtype=np.float64)
            if pop_size:
                pop[0] = best
            if pop_size > 1:
                pop[1:] = np.random.uniform(
                    zmin, zmax, size=(pop_size - 1, best.size)
                )
            return pop
        elif type == "best+uniform_on_bin_lb_ub":
            best = np.asarray(self.z0.cpu().numpy(), dtype=np.float64).reshape(-1)
            K = best.size
            pop = np.empty((pop_size, K), dtype=np.float64)
            if pop_size:
                pop[0] = best
            if pop_size > 1:
                pop[1:] = np.random.uniform(self.bin_lb, self.bin_ub, size=(pop_size - 1, K))
            return pop
        else:
            raise ValueError(f"Invalid type: {type}")

class LayerwiseScaledRandomProjection(ParameterAdapterBase):
    """Layer-wise random projection with per-layer alpha and direct biases.

    Each non-bias tensor ``l`` has a projection matrix ``P_l in R^{n_l x k}`` and
    a dedicated scalar ``alpha_l``:

        delta_l = alpha_l * (P_l @ z_l)

    Latent dimensionality is:

        num_dims = (k + 1) * L + sum(bias_sizes)

    where ``L`` is the number of projected (non-bias) parameter tensors.
    Bias tensors are evolved directly (no projection).
    """

    def __init__(self, model, k, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = len(self.base_params)
        self.k = k
        self.device = device

        self._rng = torch.Generator()
        self._rng.manual_seed(int(seed))
        torch.manual_seed(int(seed))

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        self.projections = []
        total_dims = 0
        n_projected_layers = 0
        n_bias_dims = 0

        for name, param in self.model.named_parameters():
            shape = param.shape
            size = int(math.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)

            if name.endswith(".bias"):
                dims = size
                self.layer_info.append(
                    {
                        "type": "direct",
                        "name": name,
                        "n": size,
                        "dims": dims,
                        "offset": total_dims,
                        "size": size,
                    }
                )
                total_dims += dims
                n_bias_dims += dims
            else:
                P = (
                    torch.randn(size, self.k, generator=self._rng, dtype=torch.float64)
                    / math.sqrt(self.k)
                )
                proj_idx = len(self.projections)
                self.projections.append(P)
                dims = self.k + 1  # k latent coords + 1 alpha scale
                self.layer_info.append(
                    {
                        "type": "proj_scaled",
                        "name": name,
                        "dims": dims,
                        "offset": total_dims,
                        "size": size,
                        "proj_idx": proj_idx,
                        "k": self.k,
                    }
                )
                total_dims += dims
                n_projected_layers += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"LayerwiseScaledRandomProj: L={n_projected_layers}, k={self.k}, "
            f"alpha_dims={n_projected_layers}, bias_dims={n_bias_dims} | "
            f"z dim = {self.num_dims} | model params = {self.N}"
        )

    def _init_z0(self):
        z0 = torch.zeros(self.num_dims, dtype=torch.float64, device=torch.device("cpu"))
        for info in self.layer_info:
            if info["type"] == "proj_scaled":
                z0[info["offset"] + self.k] = 1.0
        return z0

    def decode(self, z):
        """Expand latent vector to full parameter-space delta."""
        z = _z_to_torch_cpu(z)

        reconstructed = []
        for info in self.layer_info:
            offset = info["offset"]
            dims = info["dims"]
            z_layer = z[offset : offset + dims]

            if info["type"] == "proj_scaled":
                z_lat = z_layer[: self.k]
                alpha = float(z_layer[self.k].item())
                P = self.projections[info["proj_idx"]]
                reconstructed.append(alpha * (P @ z_lat))
            else:
                reconstructed.append(z_layer.clone())

        return torch.cat(reconstructed)

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Apply ``alpha`` scaling to the decoded flat vector."""
        return _to_device_float32(x, self.device) * alpha






class FlattenLoRA(ParameterAdapterBase):
    """LoRA-style low-rank decomposition over flattened model parameters.

    For each 2D or 4D weight, ``z`` holds ``A`` and ``B`` (``m*r + n*r`` values)
    plus one extra scalar **alpha_layer** that scales that tensor's LoRA delta
    ``A @ B.T`` (in addition to the global ``process(..., alpha=...)`` scalar).
    """

    def __init__(self, model, r, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = len(self.base_params)
        self.r = r
        self.device = device

        torch.manual_seed(int(seed))

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        total_dims = 0
        dims_conv = 0
        dims_linear = 0
        dims_other = 0
        module_for_param = _build_param_module_map(model)

        for name, param in model.named_parameters():
            shape = param.shape
            self.param_shapes.append(shape)
            mod_type = module_for_param.get(name)
            is_conv2d = mod_type is not None and issubclass(mod_type, nn.Conv2d)
            is_linear = mod_type is not None and issubclass(mod_type, nn.Linear)

            if len(shape) == 2:
                m, n = shape
                dims = m * r + n * r + 1
                self.layer_info.append({'type': '2d', 'm': m, 'n': n, 'dims': dims, 'offset': total_dims})
                total_dims += dims
                if is_linear:
                    dims_linear += dims
                else:
                    dims_other += dims
            elif len(shape) == 1:
                n = shape[0]
                dims = n
                self.layer_info.append({'type': '1d', 'n': n, 'dims': dims, 'offset': total_dims})
                total_dims += dims
                if is_linear:
                    dims_linear += dims
                else:
                    dims_other += dims
            elif len(shape) == 4:
                out_ch, in_ch, h, w = shape
                m, n = out_ch, in_ch * h * w
                dims = m * r + n * r + 1
                self.layer_info.append({
                    'type': '4d', 'm': m, 'n': n, 'original_shape': shape,
                    'dims': dims, 'offset': total_dims
                })
                total_dims += dims
                if is_conv2d:
                    dims_conv += dims
                else:
                    dims_other += dims
            else:
                n = int(math.prod(shape))
                dims = n
                self.layer_info.append({'type': 'other', 'n': n, 'original_shape': shape, 'dims': dims, 'offset': total_dims})
                total_dims += dims
                dims_other += dims

            self.param_sizes.append(int(math.prod(shape)))

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        extra = f", other={dims_other}" if dims_other else ""
        print(
            f"FlattenLoRA: z dim = {total_dims} (conv={dims_conv}, linear={dims_linear}{extra}) | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        z0 = torch.zeros(self.num_dims, dtype=torch.float64, device=torch.device("cpu"))
        for layer_info in self.layer_info:
            if layer_info['type'] in ('2d', '4d'):
                z0[layer_info['offset'] + layer_info['dims'] - 1] = 1.0
        return z0

    def decode(self, z):
        """Expand low-rank vector z to full parameter space."""
        z = _z_to_torch_cpu(z)

        reconstructed_params = []
        for layer_info in self.layer_info:
            offset = layer_info['offset']
            dims = layer_info['dims']
            z_layer = z[offset:offset + dims]

            if layer_info['type'] == '2d':
                m, n, r = layer_info['m'], layer_info['n'], self.r
                A_flat = z_layer[:m * r]
                B_flat = z_layer[m * r : m * r + n * r]
                alpha_layer = float(z_layer[m * r + n * r].item())
                A = A_flat.reshape(m, r)
                B = B_flat.reshape(n, r)
                W = (A @ B.T) * alpha_layer
                reconstructed_params.append(W.flatten())
            elif layer_info['type'] == '1d':
                reconstructed_params.append(z_layer[: layer_info['n']])
            elif layer_info['type'] == '4d':
                m, n = layer_info['m'], layer_info['n']
                original_shape = layer_info['original_shape']
                r = self.r
                A_flat = z_layer[:m * r]
                B_flat = z_layer[m * r : m * r + n * r]
                alpha_layer = float(z_layer[m * r + n * r].item())
                A = A_flat.reshape(m, r)
                B = B_flat.reshape(n, r)
                W_2d = A @ B.T
                W = W_2d.reshape(original_shape) * alpha_layer
                reconstructed_params.append(W.flatten())
            elif layer_info['type'] == 'other':
                n = layer_info['n']
                reconstructed_params.append(z_layer[:n] * (1.0 / math.sqrt(n)))

        return torch.cat(reconstructed_params)

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Convert to tensor."""
        return _to_device_float32(x, self.device) * alpha




class DictLoRA(ParameterAdapterBase):
    """
    Dictionary-based LoRA for ConvNets.
    For Conv2d: ΔW[o,i,:,:] = α · (Σ_m A[o,m]·B[i,m]·D[m,:,:]) / sqrt(M)
    For Linear 2D: ΔW = α · (A @ B.T).  The scalar α is one extra dimension per
    LoRA block in ``z`` (last entry of that block), in addition to global ``process`` α.
    For Linear/bias 1D: direct (no per-layer α).
    """

    def __init__(self, model, r, device="cuda", seed=42):
        self.model = model
        self.base_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.N = len(self.base_params)
        self.r = r
        self.device = device

        torch.manual_seed(int(seed))

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        base_offset = 0
        total_dims = 0
        dims_conv = 0
        dims_linear = 0
        dims_other = 0
        module_for_param = _build_param_module_map(model)

        for name, param in model.named_parameters():
            shape = param.shape
            self.param_shapes.append(shape)
            size = int(math.prod(shape))
            self.param_sizes.append(size)
            mod_type = module_for_param.get(name)
            is_conv2d = mod_type is not None and issubclass(mod_type, nn.Conv2d)
            is_linear = mod_type is not None and issubclass(mod_type, nn.Linear)

            if len(shape) == 2:
                m, n = shape
                dims = m * r + n * r + 1
                self.layer_info.append({
                    'type': '2d', 'm': m, 'n': n, 'dims': dims, 'offset': total_dims,
                    'base_offset': base_offset, 'base_size': size,
                })
                total_dims += dims
                if is_linear:
                    dims_linear += dims
                else:
                    dims_other += dims
            elif len(shape) == 1:
                n = shape[0]
                dims = n
                self.layer_info.append({
                    'type': '1d', 'n': n, 'dims': dims, 'offset': total_dims,
                    'base_offset': base_offset, 'base_size': size,
                })
                total_dims += dims
                if is_linear:
                    dims_linear += dims
                else:
                    dims_other += dims
            elif len(shape) == 4:
                out_ch, in_ch, kh, kw = shape
                M = self.r
                dims = M * (out_ch + in_ch + kh * kw) + 1
                self.layer_info.append({
                    'type': '4d_dict',
                    'Cout': out_ch, 'Cin': in_ch, 'kh': kh, 'kw': kw,
                    'M': M, 'dims': dims, 'offset': total_dims,
                    'original_shape': shape, 'base_offset': base_offset, 'base_size': size,
                })
                total_dims += dims
                if is_conv2d:
                    dims_conv += dims
                else:
                    dims_other += dims
            else:
                n = size
                dims = n
                self.layer_info.append({
                    'type': 'other', 'n': n, 'original_shape': shape, 'dims': dims,
                    'offset': total_dims, 'base_offset': base_offset, 'base_size': size,
                })
                total_dims += dims
                dims_other += dims

            base_offset += size

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        extra = f", other={dims_other}" if dims_other else ""
        print(
            f"DictLoRA: z dim = {total_dims} (conv={dims_conv}, linear={dims_linear}{extra}) | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        z0 = torch.zeros(self.num_dims, dtype=torch.float64, device=torch.device("cpu"))
        for layer_info in self.layer_info:
            if layer_info['type'] in ('2d', '4d_dict'):
                z0[layer_info['offset'] + layer_info['dims'] - 1] = 1.0
        return z0

    def _expand_4d_dict(self, z_layer, Cout, Cin, kh, kw, M):
        """Compute ΔW from DictLoRA factors A, B, D."""
        A_flat = z_layer[:Cout * M]
        B_flat = z_layer[Cout * M:Cout * M + Cin * M]
        D_flat = z_layer[Cout * M + Cin * M:]
        A = A_flat.reshape(Cout, M)
        B = B_flat.reshape(Cin, M)
        D = D_flat.reshape(M, kh, kw)
        delta_W = torch.einsum("om,im,mhw->oihw", A, B, D)
        return delta_W * (1.0 / math.sqrt(M))

    def decode(self, z):
        """Expand latent vector to full parameter space (deltas for 4d_dict)."""
        z = _z_to_torch_cpu(z)

        reconstructed = []
        for layer_info in self.layer_info:
            offset = layer_info['offset']
            dims = layer_info['dims']
            z_layer = z[offset:offset + dims]

            if layer_info['type'] == '2d':
                m, n, r = layer_info['m'], layer_info['n'], self.r
                A_flat = z_layer[:m * r]
                B_flat = z_layer[m * r : m * r + n * r]
                alpha_layer = float(z_layer[m * r + n * r].item())
                A = A_flat.reshape(m, r)
                B = B_flat.reshape(n, r)
                W = (A @ B.T) * alpha_layer
                reconstructed.append(W.flatten())
            elif layer_info['type'] == '1d':
                reconstructed.append(z_layer[: layer_info['n']])
            elif layer_info['type'] == '4d_dict':
                alpha_layer = float(z_layer[-1].item())
                delta_W = self._expand_4d_dict(
                    z_layer[:-1],
                    layer_info['Cout'], layer_info['Cin'],
                    layer_info['kh'], layer_info['kw'],
                    layer_info['M'],
                )
                reconstructed.append((delta_W * alpha_layer).flatten())
            elif layer_info['type'] == 'other':
                n = layer_info['n']
                reconstructed.append(z_layer[:n] * (1.0 / math.sqrt(n)))

        return torch.cat(reconstructed)

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Return delta only; caller adds base_params."""
        return _to_device_float32(x, self.device) * alpha




def _build_param_module_map(model):
    """Map each parameter's full name to its immediate parent module type."""
    mapping = {}
    for mod_name, mod in model.named_modules():
        for pname, _ in mod.named_parameters(recurse=False):
            full_name = f"{mod_name}.{pname}" if mod_name else pname
            mapping[full_name] = type(mod)
    return mapping


class LinearOnlyLoRA(ParameterAdapterBase):
    """LoRA exclusively on ``nn.Linear`` layers; everything else frozen.

    Conv2d, BatchNorm, LayerNorm, and all other parameters produce zero delta.
    Linear weights get rank-*r* LoRA plus one scalar **alpha_layer** per weight
    (last entry of each LoRA block in ``z``); Linear biases are evolved directly.

    Ideal for architectures where class-discriminative information concentrates
    in linear projections (transformer Q/V/MLP, classifier FC heads).
    """

    def __init__(self, model, r, device="cuda", seed=42):
        self.model = model
        self.r = r
        self.device = device

        torch.manual_seed(int(seed))

        self.base_params = torch.nn.utils.parameters_to_vector(model.parameters())
        self.N = len(self.base_params)

        module_for_param = _build_param_module_map(model)

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        total_dims = 0
        n_lora = 0
        n_direct = 0
        n_frozen = 0
        dims_conv = 0
        dims_linear = 0

        for name, param in model.named_parameters():
            shape = param.shape
            size = int(math.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)
            mod_type = module_for_param.get(name)
            is_linear = mod_type is not None and issubclass(mod_type, nn.Linear)

            if is_linear and len(shape) == 2:
                m, n = shape
                dims = m * r + n * r + 1
                self.layer_info.append({
                    'type': 'lora_2d', 'name': name, 'm': m, 'n': n,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_lora += 1
            elif is_linear and len(shape) == 1:
                dims = shape[0]
                self.layer_info.append({
                    'type': 'direct_1d', 'name': name, 'n': dims,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_direct += 1
            else:
                self.layer_info.append({
                    'type': 'frozen', 'name': name,
                    'dims': 0, 'offset': total_dims, 'size': size,
                })
                n_frozen += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"LinearOnlyLoRA: {n_lora} LoRA layers, {n_direct} direct (bias), "
            f"{n_frozen} frozen | z dim = {total_dims} (conv={dims_conv}, linear={dims_linear}) | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        z0 = torch.zeros(self.num_dims, dtype=torch.float64, device=torch.device("cpu"))
        for info in self.layer_info:
            if info['type'] == 'lora_2d':
                z0[info['offset'] + info['dims'] - 1] = 1.0
        return z0

    def decode(self, z):
        """Expand z to a full-length parameter delta (zeros for frozen layers)."""
        z = _z_to_torch_cpu(z)

        parts = []
        for info in self.layer_info:
            if info['type'] == 'lora_2d':
                offset = info['offset']
                m, n, r = info['m'], info['n'], self.r
                A = z[offset:offset + m * r].reshape(m, r)
                B = z[offset + m * r : offset + m * r + n * r].reshape(n, r)
                alpha_layer = float(z[offset + m * r + n * r].item())
                parts.append(((A @ B.T) * alpha_layer).flatten())
            elif info['type'] == 'direct_1d':
                offset = info['offset']
                parts.append(z[offset:offset + info['n']].clone())
            else:
                parts.append(torch.zeros(info['size'], dtype=torch.float64, device=torch.device("cpu")))

        return torch.cat(parts)

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        x = _to_device_float32(x, self.device)
        return alpha * x.to(self.device)




class ModulationLoRA(ParameterAdapterBase):
    """Per-channel multiplicative scaling for Conv2d + LoRA for Linear layers.

    Conv2d weights: each output channel gets a single scale factor gamma.
        ``delta_W[o,:,:,:] = gamma[o] * W_base[o,:,:,:]``
        gamma=0 at init (z0=0) produces zero delta; the base model is preserved.

    Linear weights: rank-*r* LoRA with one **alpha_layer** per linear weight
    (last entry of each LoRA block in ``z``).
    Linear biases: evolved directly.
    BatchNorm, LayerNorm, and all other parameters: frozen (zero delta).

    The conv modulation is strictly per-channel, so it cannot destroy the
    spatial filter structure -- it can only amplify or suppress channels.
    """

    def __init__(self, model, r, device="cuda", seed=42):
        self.model = model
        self.r = r
        self.device = device

        torch.manual_seed(int(seed))

        self.base_params = torch.nn.utils.parameters_to_vector(model.parameters())
        self.N = len(self.base_params)

        module_for_param = _build_param_module_map(model)

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        total_dims = 0
        n_modulated = 0
        n_lora = 0
        n_direct = 0
        n_frozen = 0
        dims_conv = 0
        dims_linear = 0

        for name, param in model.named_parameters():
            shape = param.shape
            size = int(math.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)
            mod_type = module_for_param.get(name)
            is_conv2d = mod_type is not None and issubclass(mod_type, nn.Conv2d)
            is_linear = mod_type is not None and issubclass(mod_type, nn.Linear)

            if is_conv2d and len(shape) == 4:
                Cout = shape[0]
                dims = Cout
                base_w = param.detach().cpu().clone()
                self.layer_info.append({
                    'type': 'modulation_4d', 'name': name, 'Cout': Cout,
                    'dims': dims, 'offset': total_dims, 'size': size,
                    'shape': shape, 'base_weight': base_w,
                })
                total_dims += dims
                dims_conv += dims
                n_modulated += 1
            elif is_linear and len(shape) == 2:
                m, n = shape
                dims = m * r + n * r + 1
                self.layer_info.append({
                    'type': 'lora_2d', 'name': name, 'm': m, 'n': n,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_lora += 1
            elif is_linear and len(shape) == 1:
                dims = shape[0]
                self.layer_info.append({
                    'type': 'direct_1d', 'name': name, 'n': dims,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_direct += 1
            else:
                self.layer_info.append({
                    'type': 'frozen', 'name': name,
                    'dims': 0, 'offset': total_dims, 'size': size,
                })
                n_frozen += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"ModulationLoRA: {n_modulated} modulated conv, {n_lora} LoRA linear, "
            f"{n_direct} direct (bias), {n_frozen} frozen | "
            f"z dim = {total_dims} (conv={dims_conv}, linear={dims_linear}) | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        z0 = torch.zeros(self.num_dims, dtype=torch.float64, device=torch.device("cpu"))
        for info in self.layer_info:
            if info['type'] == 'lora_2d':
                z0[info['offset'] + info['dims'] - 1] = 1.0
        return z0

    def decode(self, z):
        """Expand z to a full-length parameter delta.

        Conv deltas are multiplicative: ``gamma * base_weight``.
        LoRA deltas are additive: ``alpha_layer * (A @ B^T)``.
        """
        z = _z_to_torch_cpu(z)

        parts = []
        for info in self.layer_info:
            if info['type'] == 'modulation_4d':
                offset = info['offset']
                Cout = info['Cout']
                gamma = z[offset:offset + Cout]
                base_w = info['base_weight']
                delta = gamma.reshape(Cout, 1, 1, 1).to(base_w.dtype) * base_w
                parts.append(delta.flatten())
            elif info['type'] == 'lora_2d':
                offset = info['offset']
                m, n, r = info['m'], info['n'], self.r
                A = z[offset:offset + m * r].reshape(m, r)
                B = z[offset + m * r : offset + m * r + n * r].reshape(n, r)
                alpha_layer = float(z[offset + m * r + n * r].item())
                parts.append(((A @ B.T) * alpha_layer).flatten())
            elif info['type'] == 'direct_1d':
                offset = info['offset']
                parts.append(z[offset:offset + info['n']].clone())
            else:
                parts.append(torch.zeros(info['size'], dtype=torch.float64, device=torch.device("cpu")))

        return torch.cat(parts)

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        x = _to_device_float32(x, self.device)
        return alpha * x.to(self.device)




class SpectralLoRA(ParameterAdapterBase):
    """SVD spectral modulation for Conv2d + LoRA for Linear layers.

    Each Conv2d weight is reshaped to 2D and decomposed via SVD at init::

        W_2d = U @ diag(sigma) @ V^T

    The frozen singular vectors ``U[:, :k]`` and ``V[:, :k]`` define the
    eigenbasis; evolution modulates the top-*k* singular values only::

        delta_W_2d = U[:, :k] @ diag(z_layer) @ V[:, :k]^T

    The mapping z -> delta is **linear**: each z_i independently controls
    one orthogonal mode of the weight matrix.  Spatial filter structure is
    completely preserved because U and V are frozen from the base model.

    Linear weights: rank-*r* LoRA with one **alpha_layer** scalar per linear weight
    (last entry of each LoRA block in ``z``).
    Linear biases: evolved directly.
    BatchNorm, LayerNorm, and all other parameters: frozen (zero delta).

    ``lora_rank`` controls both the number of SVD modes *k* for conv layers
    and the LoRA rank *r* for linear layers.
    """

    def __init__(self, model, r, device="cuda", seed=42):
        self.model = model
        self.r = r
        self.device = device

        torch.manual_seed(int(seed))

        self.base_params = torch.nn.utils.parameters_to_vector(model.parameters())
        self.N = len(self.base_params)

        module_for_param = _build_param_module_map(model)

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        total_dims = 0
        n_spectral = 0
        n_lora = 0
        n_direct = 0
        n_frozen = 0
        dims_conv = 0
        dims_linear = 0

        for name, param in model.named_parameters():
            shape = param.shape
            size = int(math.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)
            mod_type = module_for_param.get(name)
            is_conv2d = mod_type is not None and issubclass(mod_type, nn.Conv2d)
            is_linear = mod_type is not None and issubclass(mod_type, nn.Linear)

            if is_conv2d and len(shape) == 4:
                Cout, Cin, kh, kw = shape
                W_2d = param.detach().cpu().reshape(Cout, Cin * kh * kw).float()
                full_rank = min(Cout, Cin * kh * kw)
                k = min(r, full_rank)
                U_full, sigma, Vh_full = torch.linalg.svd(W_2d, full_matrices=False)
                U_k = U_full[:, :k].clone()
                Vt_k = Vh_full[:k, :].clone()

                self.layer_info.append({
                    'type': 'spectral_4d', 'name': name,
                    'k': k, 'dims': k, 'offset': total_dims, 'size': size,
                    'shape': shape, 'U_k': U_k, 'Vt_k': Vt_k,
                    'sigma_k': sigma[:k].clone(),
                })
                total_dims += k
                dims_conv += k
                n_spectral += 1
            elif is_linear and len(shape) == 2:
                m, n = shape
                dims = m * r + n * r + 1
                self.layer_info.append({
                    'type': 'lora_2d', 'name': name, 'm': m, 'n': n,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_lora += 1
            elif is_linear and len(shape) == 1:
                dims = shape[0]
                self.layer_info.append({
                    'type': 'direct_1d', 'name': name, 'n': dims,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_direct += 1
            else:
                self.layer_info.append({
                    'type': 'frozen', 'name': name,
                    'dims': 0, 'offset': total_dims, 'size': size,
                })
                n_frozen += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"SpectralLoRA: {n_spectral} spectral conv (k={r}), {n_lora} LoRA linear, "
            f"{n_direct} direct (bias), {n_frozen} frozen | "
            f"z dim = {total_dims} (conv={dims_conv}, linear={dims_linear}) | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        z0 = torch.zeros(self.num_dims, dtype=torch.float64, device=torch.device("cpu"))
        for info in self.layer_info:
            if info['type'] == 'lora_2d':
                z0[info['offset'] + info['dims'] - 1] = 1.0
        return z0

    def decode(self, z):
        """Expand z to a full-length parameter delta.

        Conv deltas via spectral modulation: ``U_k @ diag(z) @ Vt_k``.
        Linear deltas via LoRA: ``alpha_layer * (A @ B^T)``.
        """
        z = _z_to_torch_cpu(z)

        parts = []
        for info in self.layer_info:
            if info['type'] == 'spectral_4d':
                offset = info['offset']
                k = info['k']
                z_sv = z[offset:offset + k]
                U_k = info['U_k'].to(z_sv.dtype)
                Vt_k = info['Vt_k'].to(z_sv.dtype)
                delta_2d = (U_k * z_sv) @ Vt_k
                parts.append(delta_2d.flatten())
            elif info['type'] == 'lora_2d':
                offset = info['offset']
                m, n, r = info['m'], info['n'], self.r
                A = z[offset:offset + m * r].reshape(m, r)
                B = z[offset + m * r : offset + m * r + n * r].reshape(n, r)
                alpha_layer = float(z[offset + m * r + n * r].item())
                parts.append(((A @ B.T) * alpha_layer).flatten())
            elif info['type'] == 'direct_1d':
                offset = info['offset']
                parts.append(z[offset:offset + info['n']].clone())
            else:
                parts.append(torch.zeros(info['size'], dtype=torch.float64, device=torch.device("cpu")))

        return torch.cat(parts)

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        x = _to_device_float32(x, self.device)
        return alpha * x.to(self.device)




class SpectralAllSVD(ParameterAdapterBase):
    """SVD spectral modulation for Conv2d and Linear 2D weights (no LoRA).

    Conv2d and ``nn.Linear`` weight matrices are treated the same: each is
    decomposed at init as ``W = U @ diag(sigma) @ V^T``; evolution only
    modulates the top-*k* singular values with frozen *U*, *V*::

        delta_W = U[:, :k] @ diag(z_layer) @ V[:, :k]^T

    Linear biases are evolved directly. BatchNorm, LayerNorm, and all other
    parameters stay frozen (zero delta).

    ``r`` sets *k* = ``min(r, rank(W))`` per layer for both conv and linear
    weights.
    """

    def __init__(self, model, r, device="cuda", seed=42):
        self.model = model
        self.r = r
        self.device = device

        torch.manual_seed(int(seed))

        self.base_params = torch.nn.utils.parameters_to_vector(model.parameters())
        self.N = len(self.base_params)

        module_for_param = _build_param_module_map(model)

        self.layer_info = []
        self.param_shapes = []
        self.param_sizes = []
        total_dims = 0
        n_spectral_conv = 0
        n_spectral_linear = 0
        n_direct = 0
        n_frozen = 0
        dims_conv = 0
        dims_linear = 0

        for name, param in model.named_parameters():
            shape = param.shape
            size = int(math.prod(shape))
            self.param_shapes.append(shape)
            self.param_sizes.append(size)
            mod_type = module_for_param.get(name)
            is_conv2d = mod_type is not None and issubclass(mod_type, nn.Conv2d)
            is_linear = mod_type is not None and issubclass(mod_type, nn.Linear)

            if is_conv2d and len(shape) == 4:
                Cout, Cin, kh, kw = shape
                W_2d = param.detach().cpu().reshape(Cout, Cin * kh * kw).float()
                full_rank = min(Cout, Cin * kh * kw)
                k = min(r, full_rank)
                U_full, sigma, Vh_full = torch.linalg.svd(W_2d, full_matrices=False)
                U_k = U_full[:, :k].clone()
                Vt_k = Vh_full[:k, :].clone()

                self.layer_info.append({
                    'type': 'spectral_4d', 'name': name,
                    'k': k, 'dims': k, 'offset': total_dims, 'size': size,
                    'shape': shape, 'U_k': U_k, 'Vt_k': Vt_k,
                    'sigma_k': sigma[:k].clone(),
                })
                total_dims += k
                dims_conv += k
                n_spectral_conv += 1
            elif is_linear and len(shape) == 2:
                W_2d = param.detach().cpu().float()
                m, n = W_2d.shape
                full_rank = min(m, n)
                k = min(r, full_rank)
                U_full, sigma, Vh_full = torch.linalg.svd(W_2d, full_matrices=False)
                U_k = U_full[:, :k].clone()
                Vt_k = Vh_full[:k, :].clone()

                self.layer_info.append({
                    'type': 'spectral_2d', 'name': name,
                    'k': k, 'dims': k, 'offset': total_dims, 'size': size,
                    'shape': shape, 'U_k': U_k, 'Vt_k': Vt_k,
                    'sigma_k': sigma[:k].clone(),
                })
                total_dims += k
                dims_linear += k
                n_spectral_linear += 1
            elif is_linear and len(shape) == 1:
                dims = shape[0]
                self.layer_info.append({
                    'type': 'direct_1d', 'name': name, 'n': dims,
                    'dims': dims, 'offset': total_dims, 'size': size,
                })
                total_dims += dims
                dims_linear += dims
                n_direct += 1
            else:
                self.layer_info.append({
                    'type': 'frozen', 'name': name,
                    'dims': 0, 'offset': total_dims, 'size': size,
                })
                n_frozen += 1

        self.num_dims = total_dims
        self.z0 = self._init_z0()

        print(
            f"SpectralAllSVD: {n_spectral_conv} spectral conv + {n_spectral_linear} spectral linear "
            f"(k<={r}), {n_direct} direct (bias), {n_frozen} frozen | "
            f"z dim = {total_dims} (conv={dims_conv}, linear={dims_linear}) | "
            f"model params = {self.N}"
        )

    def _init_z0(self):
        return torch.zeros(self.num_dims, dtype=torch.float64, device=torch.device("cpu"))

    def decode(self, z):
        """Deltas via ``U_k @ diag(z) @ Vt_k`` for conv and linear weights."""
        z = _z_to_torch_cpu(z)

        parts = []
        for info in self.layer_info:
            if info['type'] in ('spectral_4d', 'spectral_2d'):
                offset = info['offset']
                k = info['k']
                z_sv = z[offset:offset + k]
                U_k = info['U_k'].to(z_sv.dtype)
                Vt_k = info['Vt_k'].to(z_sv.dtype)
                delta_2d = (U_k * z_sv) @ Vt_k
                parts.append(delta_2d.flatten())
            elif info['type'] == 'direct_1d':
                offset = info['offset']
                parts.append(z[offset:offset + info['n']].clone())
            else:
                parts.append(torch.zeros(info['size'], dtype=torch.float64, device=torch.device("cpu")))

        return torch.cat(parts)

    def scale(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        x = _to_device_float32(x, self.device)
        return alpha * x.to(self.device)


def get_adapter(adapter_type: str, model: nn.Module, args: argparse.Namespace) -> ParameterAdapterBase:
    if adapter_type == "full":
        return FullSpace(model, device=args.device, seed=args.seed)
    elif adapter_type == "random_projection":
        return RandomProjection(model, args.adapter_k, device=args.device, seed=args.seed)
    elif adapter_type == "layerwise_random_projection":
        return LayerwiseRandomProjection(model, args.adapter_k, device=args.device, seed=args.seed)
    elif adapter_type == "layerwise_random_blocking":
        return LayerwiseRandomBlocking(model, args.adapter_k, device=args.device, seed=args.seed)
    elif adapter_type == "global_uniform_binning_with_delta":
        return GlobalUniformBinningWithDelta(model, args.adapter_k, device=args.device, seed=args.seed)
    elif adapter_type == "global_uniform_binning_directly":
        return GlobalUniformBinningDirectly(model, args.adapter_k, device=args.device, seed=args.seed)
    elif adapter_type == "layerwise_scaled_random_projection":
        return LayerwiseScaledRandomProjection(model, args.adapter_k, device=args.device, seed=args.seed)
    elif adapter_type == "flatten_lora":
        return FlattenLoRA(model, args.adapter_rank, device=args.device, seed=args.seed)
    elif adapter_type == "dict_lora":
        return DictLoRA(model, args.adapter_rank, device=args.device, seed=args.seed)
    elif adapter_type == "linear_only_lora":
        return LinearOnlyLoRA(model, args.adapter_rank, device=args.device, seed=args.seed)
    elif adapter_type == "modulation_lora":
        return ModulationLoRA(model, args.adapter_rank, device=args.device, seed=args.seed)
    elif adapter_type == "spectral_lora":
        return SpectralLoRA(model, args.adapter_k, device=args.device, seed=args.seed)
    elif adapter_type == "spectral_all_svd":
        return SpectralAllSVD(model, args.adapter_k, device=args.device, seed=args.seed)
    else:
        raise ValueError(f"Invalid adapter type: {adapter_type}")