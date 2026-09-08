"""Parallel Associative Prefix Scan for Pseudo-Brain.

Breaks the sequential recurrence wall by reformulating the linear recurrence
part of recurrent neural state transitions as an associative binary operator:

    h_t = a_t * h_{t-1} + b_t

For two consecutive affine state transitions:
    f_1(h) = a_1 * h + b_1
    f_2(h) = a_2 * h + b_2
The composition (f_2 o f_1)(h) is:
    f_2(f_1(h)) = a_2 * (a_1 * h + b_1) + b_2
                = (a_1 * a_2) * h + (a_2 * b_1 + b_2)

Hence, the associative binary operator is defined as:
    (a_1, b_1) • (a_2, b_2) = (a_1 * a_2, a_2 * b_1 + b_2)

Associativity Proof:
    Let x = (a_1, b_1), y = (a_2, b_2), z = (a_3, b_3).
    (x • y) • z = (a_1 * a_2, a_2 * b_1 + b_2) • (a_3, b_3)
                = ((a_1 * a_2) * a_3, a_3 * (a_2 * b_1 + b_2) + b_3)
                = (a_1 * a_2 * a_3, a_3 * a_2 * b_1 + a_3 * b_2 + b_3)

    x • (y • z) = (a_1, b_1) • (a_2 * a_3, a_3 * b_2 + b_3)
                = (a_1 * (a_2 * a_3), (a_2 * a_3) * b_1 + (a_3 * b_2 + b_3))
                = (a_1 * a_2 * a_3, a_3 * a_2 * b_1 + a_3 * b_2 + b_3)

    Since (x • y) • z == x • (y • z), the operator • is strictly associative.
    Identity element: e = (1, 0), where (1, 0) • (a, b) = (a, b) • (1, 0) = (a, b).
    Therefore, ((A, B), •) forms a monoid.

By Blelloch / Kogge-Stone parallel prefix scan principles, any associative prefix
computation over a sequence of length T can be executed in O(log T) sequential
parallel steps rather than O(T) serial Python loops, reducing training latency
from O(T) to O(log T).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
from torch import Tensor


def associative_binary_op(
    a1: Tensor, b1: Tensor, a2: Tensor, b2: Tensor
) -> Tuple[Tensor, Tensor]:
    """Applies the associative binary operator (a_1, b_1) • (a_2, b_2).

    Args:
        a1: Retention / decay tensor from earlier step(s).
        b1: Candidate / value tensor from earlier step(s).
        a2: Retention / decay tensor from later step(s).
        b2: Candidate / value tensor from later step(s).

    Returns:
        Tuple of combined (a_new, b_new) where:
            a_new = a1 * a2
            b_new = a2 * b1 + b2
    """
    return a1 * a2, a2 * b1 + b2


def associative_scan_work_efficient(
    a: Tensor, b: Tensor
) -> Tuple[Tensor, Tensor]:
    """Work-efficient parallel prefix scan (recursive doubling / Blelloch tree).

    Complexity:
        - Step complexity: O(log T) sequential parallel barriers.
        - Work complexity: O(T) total elementwise operations (FLOPs).

    Handles arbitrary sequence lengths T (odd or even) and arbitrary batch/feature shapes.
    Uses purely functional out-of-place PyTorch primitives for seamless autograd backpropagation.

    Args:
        a: Decay / retention gate tensor with shape [B, T, ...] along sequence dim 1.
        b: Candidate / injection tensor with shape [B, T, ...] along sequence dim 1.

    Returns:
        Tuple of (prefix_a, prefix_b) where prefix_b contains all T recurrent states.
    """
    T = a.shape[1]
    if T <= 1:
        return a, b

    # 1. Split into even-indexed (t=0, 2, 4, ...) and odd-indexed (t=1, 3, 5, ...) elements
    a_even, b_even = a[:, 0::2], b[:, 0::2]
    a_odd, b_odd = a[:, 1::2], b[:, 1::2]

    is_odd = (T % 2 == 1)
    if is_odd:
        # Last even element does not have a matching odd partner in this reduction pass
        a_even_pair = a_even[:, :-1]
        b_even_pair = b_even[:, :-1]
    else:
        a_even_pair = a_even
        b_even_pair = b_even

    # 2. Pairwise reduction: combine adjacent pairs (even_i • odd_i)
    a_reduced, b_reduced = associative_binary_op(
        a_even_pair, b_even_pair, a_odd, b_odd
    )

    # 3. Recursive scan on reduced sequence of length floor(T / 2)
    a_scanned_odd, b_scanned_odd = associative_scan_work_efficient(
        a_reduced, b_reduced
    )

    # 4. Down-sweep: reconstruct full prefix states
    # - Odd elements are directly the scanned results of pairs:
    #   scanned_odd[k] = e_0 • e_1 • ... • e_{2k+1}
    # - Even elements (for k >= 1):
    #   scanned_even[k] = scanned_odd[k-1] • even[k]
    # - Even element 0 is unchanged: even[0]
    a_even_tail = a_even[:, 1:]
    b_even_tail = b_even[:, 1:]

    if is_odd:
        # scanned_odd has length (T - 1) / 2, matching a_even_tail length
        a_comb_even, b_comb_even = associative_binary_op(
            a_scanned_odd, b_scanned_odd, a_even_tail, b_even_tail
        )
        a_scanned_even = torch.cat([a_even[:, :1], a_comb_even], dim=1)
        b_scanned_even = torch.cat([b_even[:, :1], b_comb_even], dim=1)

        # Interleave: [even_0, odd_0, even_1, odd_1, ..., odd_{M-1}, even_M]
        out_a_pairs = torch.stack(
            [a_scanned_even[:, :-1], a_scanned_odd], dim=2
        ).flatten(1, 2)
        out_b_pairs = torch.stack(
            [b_scanned_even[:, :-1], b_scanned_odd], dim=2
        ).flatten(1, 2)
        out_a = torch.cat([out_a_pairs, a_scanned_even[:, -1:]], dim=1)
        out_b = torch.cat([out_b_pairs, b_scanned_even[:, -1:]], dim=1)
    else:
        # scanned_odd has length T / 2; a_even_tail has length T / 2 - 1
        a_comb_even, b_comb_even = associative_binary_op(
            a_scanned_odd[:, :-1], b_scanned_odd[:, :-1], a_even_tail, b_even_tail
        )
        a_scanned_even = torch.cat([a_even[:, :1], a_comb_even], dim=1)
        b_scanned_even = torch.cat([b_even[:, :1], b_comb_even], dim=1)

        # Interleave: [even_0, odd_0, even_1, odd_1, ...]
        out_a = torch.stack([a_scanned_even, a_scanned_odd], dim=2).flatten(1, 2)
        out_b = torch.stack([b_scanned_even, b_scanned_odd], dim=2).flatten(1, 2)

    return out_a, out_b


def associative_scan_hillis_steele(
    a: Tensor, b: Tensor
) -> Tuple[Tensor, Tensor]:
    """Step-efficient parallel prefix scan (Hillis-Steele algorithm).

    Complexity:
        - Step complexity: ceil(log2 T) sequential parallel steps.
        - Work complexity: O(T log T) operations.

    Args:
        a: Decay / retention gate tensor with shape [B, T, ...] along sequence dim 1.
        b: Candidate / injection tensor with shape [B, T, ...] along sequence dim 1.

    Returns:
        Tuple of (prefix_a, prefix_b) where prefix_b contains all T recurrent states.
    """
    T = a.shape[1]
    if T <= 1:
        return a, b

    num_steps = (T - 1).bit_length()
    cur_a = a
    cur_b = b

    for step in range(num_steps):
        d = 1 << step
        left_a = cur_a[:, :-d]
        left_b = cur_b[:, :-d]
        right_a = cur_a[:, d:]
        right_b = cur_b[:, d:]

        comb_a, comb_b = associative_binary_op(left_a, left_b, right_a, right_b)
        cur_a = torch.cat([cur_a[:, :d], comb_a], dim=1)
        cur_b = torch.cat([cur_b[:, :d], comb_b], dim=1)

    return cur_a, cur_b


def sequential_scan(
    a: Tensor,
    b: Tensor,
    h0: Optional[Tensor] = None,
    dim: int = 1,
) -> Tensor:
    """Reference sequential loop implementation of the recurrence h_t = a_t * h_{t-1} + b_t.

    Used for numerical verification, testing, and baseline speed comparisons.

    Args:
        a: Decay / retention gate tensor.
        b: Candidate / injection tensor.
        h0: Optional initial recurrent state.
        dim: Sequence dimension along which to scan.

    Returns:
        Hidden states tensor of shape matching b.
    """
    if dim != 1:
        a = a.transpose(1, dim)
        b = b.transpose(1, dim)

    B, T = b.shape[:2]
    if T == 0:
        return b.clone()

    if h0 is not None:
        h = h0
    else:
        h = torch.zeros_like(b[:, 0])

    states = []
    for t in range(T):
        h = a[:, t] * h + b[:, t]
        states.append(h)

    out = torch.stack(states, dim=1)

    if dim != 1:
        out = out.transpose(1, dim)
    return out


def parallel_scan(
    a: Tensor,
    b: Tensor,
    h0: Optional[Tensor] = None,
    dim: int = 1,
    method: Literal["work_efficient", "hillis_steele", "sequential"] = "work_efficient",
    h_init: Optional[Tensor] = None,
) -> Tensor:
    if h_init is not None and h0 is None:
        h0 = h_init
    """Computes the linear recurrence h_t = a_t * h_{t-1} + b_t in parallel.

    Computes all T states in parallel in O(log T) steps instead of O(T) sequential steps.

    If an initial state h_0 is provided:
        h_1 = a_1 * h_0 + b_1
    By folding h_0 into the first candidate term b'_1 = b_1 + a_1 * h_0, the prefix scan
    directly computes the exact recurrent state sequence [h_1, h_2, ..., h_T].

    Args:
        a: Decay / retention gate tensor with values typically in [0, 1].
           Shape (..., T, ...). Broadcastable to b.
        b: Candidate / injection tensor of shape (..., T, ...).
        h0: Optional initial state tensor of shape (..., ...).
        dim: Sequence dimension along which to perform the prefix scan. Default: 1.
        method: Parallel scan algorithm:
            - 'work_efficient': O(T) FLOPs, O(log T) depth (Blelloch/Brent-Kung tree). Recommended.
            - 'hillis_steele': O(T log T) FLOPs, O(log T) depth.
            - 'sequential': O(T) serial loop for testing/debugging.

    Returns:
        Hidden states tensor H of shape matching b, where H[:, t] = h_{t+1}.
    """
    if method == "sequential":
        return sequential_scan(a, b, h0=h0, dim=dim)

    # Transpose to dimension 1 if needed
    transposed = (dim != 1)
    if transposed:
        a = a.transpose(1, dim)
        b = b.transpose(1, dim)

    # Ensure a broadcasts to b if needed
    if a.shape != b.shape:
        a = a.expand_as(b)

    T = b.shape[1]
    if T == 0:
        return b.clone()

    # Incorporate initial state h0 into the first step: b'_1 = b_1 + a_1 * h_0
    if h0 is not None:
        h0_aligned = h0.unsqueeze(1) if h0.ndim == b.ndim - 1 else h0
        b0_adjusted = b[:, :1] + a[:, :1] * h0_aligned
        if T > 1:
            b_init = torch.cat([b0_adjusted, b[:, 1:]], dim=1)
        else:
            b_init = b0_adjusted
    else:
        b_init = b

    if T == 1:
        out = b_init
    elif method == "work_efficient":
        _, out = associative_scan_work_efficient(a, b_init)
    elif method == "hillis_steele":
        _, out = associative_scan_hillis_steele(a, b_init)
    else:
        raise ValueError(f"Unknown scan method: '{method}'. Choose 'work_efficient', 'hillis_steele', or 'sequential'.")

    if transposed:
        out = out.transpose(1, dim)

    return out


def parallel_scan_with_cumprod(
    a: Tensor,
    b: Tensor,
    h0: Optional[Tensor] = None,
    dim: int = 1,
    method: Literal["work_efficient", "hillis_steele"] = "work_efficient",
) -> Tuple[Tensor, Tensor]:
    """Computes parallel prefix scan returning both cumulative decays and hidden states.

    Returns:
        Tuple (cumprod_a, states_h)
            cumprod_a: Product of gates up to time t: prod_{i=1}^t a_i
            states_h: Recurrent states h_t = prod_{i=1}^t a_i * h_0 + sum_{i=1}^t (prod_{j=i+1}^t a_j) * b_i
    """
    transposed = (dim != 1)
    if transposed:
        a = a.transpose(1, dim)
        b = b.transpose(1, dim)

    if a.shape != b.shape:
        a = a.expand_as(b)

    T = b.shape[1]
    if h0 is not None:
        h0_aligned = h0.unsqueeze(1) if h0.ndim == b.ndim - 1 else h0
        b0_adjusted = b[:, :1] + a[:, :1] * h0_aligned
        b_init = torch.cat([b0_adjusted, b[:, 1:]], dim=1) if T > 1 else b0_adjusted
    else:
        b_init = b

    if T <= 1:
        cum_a, states = a, b_init
    elif method == "work_efficient":
        cum_a, states = associative_scan_work_efficient(a, b_init)
    elif method == "hillis_steele":
        cum_a, states = associative_scan_hillis_steele(a, b_init)
    else:
        raise ValueError(f"Unknown scan method: {method}")

    if transposed:
        cum_a = cum_a.transpose(1, dim)
        states = states.transpose(1, dim)

    return cum_a, states
