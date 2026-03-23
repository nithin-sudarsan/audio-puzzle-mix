"""Puzzle Mix augmentation adapted for single-channel, non-square spectrograms.

Puzzle Mix (Kim et al., 2020, ICML) constructs mixed training examples by:

  1. Computing a saliency map for each input (done in the training loop, passed in).
  2. Solving a graph-cut optimisation to find a binary spatial mask that maximises
     exposed saliency while penalising spatially incoherent cuts.
  3. Optionally applying an optimal transport step that rearranges image blocks so
     that high-saliency content moves into the regions the mask exposes.
  4. Blending the two (possibly rearranged) images using the upsampled mask.

Adaptations from the reference implementation (Kim et al., PuzzleMix/mixup.py)
-------------------------------------------------------------------------------
The reference assumes square images and a square block grid.  Our spectrograms
are (128 × 500) — non-square.  The following changes were made:

  a. Block grid: we use square pixel blocks (block_size × block_size) with
     separate row/column counts:
       block_h = H // block_size      (e.g. 128 // 32 = 4)
       block_w = valid_W // block_size (e.g. 480 // 32 = 15)
     where valid_W = (W // block_size) * block_size ensures exact integer blocks.
     The remaining columns (W - valid_W; at most 20 px for block_size=32, i.e.
     4% of W) are covered by nearest-neighbour interpolation of the block mask.

  b. Cost matrix: the transport cost matrix is computed on-the-fly for the
     (block_h × block_w) grid, with positions indexed as (row, col) and each
     axis independently normalised to [0, 1].  This replaces the reference's
     precomputed square cost_matrix_dict.

  c. Channel count: the reference hardcodes C=3 in transport_image.  We
     parametrise over the channel count (C=1 for our spectrograms).

  d. Graph cut library: the reference uses `gco.cut_grid_graph` (not available
     as a standalone package).  We use `pygco.cut_simple_vh`, which accepts
     separate vertical/horizontal pairwise weight arrays and handles rectangular
     grids natively.

  e. Adversarial component: excluded.  The adversarial perturbation in Algorithm 2
     of Kim et al. is designed to improve adversarial robustness — it is not
     relevant to the saliency faithfulness question this project investigates.
     Excluding it also keeps the Puzzle Mix condition strictly comparable to the
     other three conditions, none of which include adversarial training.

Interface contract (consistent with mixup.py and cutmix.py)
------------------------------------------------------------
    x_mixed, y_mixed = apply_puzzle_mix(x, y_onehot, saliency, ...)

The saliency map must be computed by the training loop before calling this
function.  Separating saliency computation from mixing is intentional: it means
apply_puzzle_mix has no dependence on the model, and the training loop can
compute saliency in eval mode (ensuring BN statistics do not interfere with the
gradient) before switching back to train mode for the actual update.
"""

import numpy as np
import pygco
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Transport cost matrix
# ---------------------------------------------------------------------------

def compute_cost_matrix(block_h: int, block_w: int, device: str) -> Tensor:
    """Pairwise transport cost matrix for a (block_h × block_w) block grid.

    Entry C[i, j] is the squared Euclidean distance between grid positions i
    and j, where each axis is normalised independently to [0, 1]:

        C[i, j] = ((row_i - row_j) / (block_h - 1))²
                + ((col_i - col_j) / (block_w - 1))²

    Normalising each axis to [0, 1] means a horizontal move of one full grid
    width has cost 1.0, and so does a vertical move of one full grid height.
    This is scale-invariant: a 4×15 grid has the same max distance as a 4×4
    grid, which ensures the transport regularisation weight (t_eps) has a
    consistent interpretation regardless of grid shape.

    Parameters
    ----------
    block_h, block_w : int
        Grid dimensions (number of blocks in height and width directions).
    device : str
        Torch device for the returned tensor.

    Returns
    -------
    C : Tensor, shape (1, block_h*block_w, block_h*block_w)
        Unsqueezed along dim 0 for broadcasting with batch dimensions.
    """
    n = block_h * block_w
    C = torch.zeros(n, n, dtype=torch.float32)

    denom_h = max(block_h - 1, 1)   # avoid div/0 for 1-row grids
    denom_w = max(block_w - 1, 1)

    for idx1 in range(n):
        r1, c1 = divmod(idx1, block_w)
        for idx2 in range(n):
            r2, c2 = divmod(idx2, block_w)
            C[idx1, idx2] = ((r1 - r2) / denom_h) ** 2 + ((c1 - c2) / denom_w) ** 2

    return C.unsqueeze(0).to(device)  # (1, n, n)


# ---------------------------------------------------------------------------
# Pairwise smoothness penalty
# ---------------------------------------------------------------------------

def _neigh_penalty(input1: Tensor, input2: Tensor, k_h: int, k_w: int):
    """Compute block-boundary smoothness penalties between two images.

    Measures how much the pixel values change across each block boundary,
    averaged over the pixels within each block.  High penalty → the cut would
    cross a region of high local spatial coherence → graph cut should prefer
    not to cut here.

    The inputs are pre-pooled to `neigh_size`-pixel resolution before calling
    this function (done in apply_puzzle_mix).  k_h and k_w are the number of
    neigh_size-resolution pixels per block in each direction:
        k_h = block_size // neigh_size
        k_w = block_size // neigh_size
    (equal for our square-pixel-block design, but kept separate for generality)

    Parameters
    ----------
    input1, input2 : Tensor, shape (B, C, H_pool, W_pool)
        Images pooled to neigh_size resolution.
    k_h, k_w : int
        Block size in pooled-pixel units, for height and width respectively.

    Returns
    -------
    pw_x : Tensor, shape (B, block_h - 1, block_w)
        Penalty at each horizontal row-boundary (between block row i and i+1).
    pw_y : Tensor, shape (B, block_h, block_w - 1)
        Penalty at each vertical column-boundary (between block col j and j+1).

    Implementation note: F.avg_pool2d is called on a 3D tensor (B, H, W)
    which PyTorch treats as (C=B, H, W).  This is an intentional trick from
    the original codebase: it lets us pool across the spatial dimensions while
    keeping the batch dimension intact, without needing to add/remove a channel
    dimension.
    """
    # Differences across adjacent pooled pixels in each direction
    # pw_x: differences between row i and row i+1 → shape (B, C, H_pool-1, W_pool)
    pw_x = input1[:, :, :-1, :] - input2[:, :, 1:, :]
    # pw_y: differences between col j and col j+1 → shape (B, C, H_pool, W_pool-1)
    pw_y = input1[:, :, :, :-1] - input2[:, :, :, 1:]

    # Sample at block boundaries only: every k-th pooled pixel, starting at k-1
    # For pw_x (height direction): indices k_h-1, 2*k_h-1, ... one per block row boundary
    pw_x = pw_x[:, :, k_h - 1::k_h, :]   # shape: (B, C, block_h-1, W_pool)
    pw_y = pw_y[:, :, :, k_w - 1::k_w]   # shape: (B, C, H_pool, block_w-1)

    # Average absolute difference over channels, then pool across the block width/height
    # pw_x.abs().mean(1): (B, block_h-1, W_pool)
    # avg_pool2d with (1, k_w): averages over k_w columns → (B, block_h-1, block_w)
    pw_x = F.avg_pool2d(pw_x.abs().mean(1), kernel_size=(1, k_w))
    # pw_y.abs().mean(1): (B, H_pool, block_w-1)
    # avg_pool2d with (k_h, 1): averages over k_h rows → (B, block_h, block_w-1)
    pw_y = F.avg_pool2d(pw_y.abs().mean(1), kernel_size=(k_h, 1))

    return pw_x, pw_y


# ---------------------------------------------------------------------------
# Graph cut (single example)
# ---------------------------------------------------------------------------

def _graphcut_single(
    x2_unary: np.ndarray,
    x1_unary: np.ndarray,
    pw_x: np.ndarray,
    pw_y: np.ndarray,
    alpha: float,
    beta: float,
    eta: float,
    n_labels: int = 3,
    eps: float = 1e-8,
) -> np.ndarray:
    """Solve the Puzzle Mix graph-cut mask optimisation for one example pair.

    Finds the assignment of labels to blocks that minimises:
        E = Σ_i unary_cost(i, label_i)  +  Σ_{i,j adjacent} pairwise_cost(label_i, label_j)

    where the unary term rewards exposing high-saliency blocks and the pairwise
    term penalises cuts across regions of high local coherence.

    IMPORTANT — argument order convention (matches reference implementation):
    The first argument `x2_unary` is the saliency map of the *second* source
    image (x2), and `x1_unary` is the saliency of x1.  This is because label 0
    (mask value 1.0, showing x1) has low cost when x1_unary is high:
        cost[label=0] = large_val * (x2_unary + prior[0])
        cost[label=n-1] = large_val * (x1_unary + prior[n-1])
    So a block with high x1 saliency gets high cost for label n-1 (the x2
    label), steering it toward label 0 (show x1) — which is the correct
    behaviour for saliency-guided mask optimisation.

    Parameters
    ----------
    x2_unary : ndarray, shape (block_h, block_w)
        Normalised saliency of the second source (x2), pooled to block grid.
        High values indicate discriminative regions that x2 wants to expose.
    x1_unary : ndarray, shape (block_h, block_w)
        Normalised saliency of the first source (x1).
    pw_x : ndarray, shape (block_h - 1, block_w)
        Pairwise penalty for vertical cuts (between block rows).
    pw_y : ndarray, shape (block_h, block_w - 1)
        Pairwise penalty for horizontal cuts (between block columns).
    alpha : float
        Mixing coefficient (fraction for x1).  Used only for the label prior.
    beta : float
        Pre-normalised pairwise weight (already divided by block count × 16).
    eta : float
        Prior weight.  Scales the label-prior term relative to saliency terms.
    n_labels : int
        Number of discrete mixing levels.  3 gives mask values {0, 0.5, 1.0};
        2 gives binary {0, 1}.  Matches the reference default of 3.
    eps : float
        Small constant to prevent log(0) in the prior computation.

    Returns
    -------
    mask : ndarray, shape (block_h, block_w)
        Mask values in [0, 1].  1.0 = block belongs to x1, 0.0 = belongs to x2.
    """
    block_h, block_w = x2_unary.shape
    n_blocks = block_h * block_w

    # Scale factor: large_val ensures integer approximation via int32 cast
    # does not collapse small differences.  Proportional to n_blocks to stay
    # consistent regardless of block grid size.
    large_val = 1000 * n_blocks

    # Label prior: discourages extreme mixing ratios (0 or 1) when alpha ≈ 0.5
    # For n_labels=3, linspace(0,1,3) = [0, 0.5, 1.0]
    lam_values = np.linspace(0.0, 1.0, n_labels)
    prior = np.array([-np.log(alpha * (1 - lam) + (1 - alpha) * lam + eps)
                      for lam in lam_values], dtype=np.float32)
    prior = eta * prior / n_blocks   # normalise by number of blocks

    # Unary cost for each (block, label) pair.
    # For label l with mixing coefficient lam_l:
    #   cost = large_val * ((1-lam_l)*x2_unary + lam_l*x1_unary + prior[l])
    # This yields low cost for label 0 when x1_unary is high (expose x1),
    # and low cost for label n-1 when x2_unary is high (expose x2).
    unary_cost = large_val * np.stack(
        [(1.0 - lam) * x2_unary + lam * x1_unary + prior[i]
         for i, lam in enumerate(lam_values)],
        axis=-1,
    ).astype(np.int32)   # shape: (block_h, block_w, n_labels)

    # Pairwise cost: penalises adjacent blocks with different labels.
    # (i - j)² / (n_labels - 1)² gives smooth, convex pairwise costs.
    pairwise_cost = np.array(
        [[(i - j) ** 2 / (n_labels - 1) ** 2 for j in range(n_labels)]
         for i in range(n_labels)],
        dtype=np.float32,
    )
    # Scale to int32 for pygco
    pairwise_cost_int = (large_val * pairwise_cost).astype(np.int32)

    # Pairwise edge weights.
    # pw_x (block_h-1, block_w): cost of cutting between block row i and i+1
    # pw_y (block_h, block_w-1): cost of cutting between block col j and j+1
    # Scale and add beta offset (minimum smoothness penalty everywhere)
    pw_x_scaled = (large_val * (pw_x + beta)).astype(np.int32)
    pw_y_scaled = (large_val * (pw_y + beta)).astype(np.int32)

    # Convert to pygco's costV/costH format:
    # costV[i, j] = edge cost between nodes (i, j) and (i+1, j)  → from pw_x
    # costH[i, j] = edge cost between nodes (i, j) and (i, j+1)  → from pw_y
    # Last row of costV and last col of costH are unused (no edges beyond grid)
    costV = np.zeros((block_h, block_w), dtype=np.int32)
    costH = np.zeros((block_h, block_w), dtype=np.int32)
    costV[:block_h - 1, :] = pw_x_scaled   # (block_h-1, block_w) → first block_h-1 rows
    costH[:, :block_w - 1] = pw_y_scaled   # (block_h, block_w-1) → first block_w-1 cols

    # Solve the α-β swap graph cut.
    # pygco.cut_simple_vh minimises the total energy (unary + pairwise) over
    # the grid graph, using the α-β swap algorithm for multi-label problems.
    # Returns an integer label array of shape (block_h, block_w).
    labels = pygco.cut_simple_vh(
        unary_cost, pairwise_cost_int, costV, costH,
        n_iter=5, algorithm='swap',
    )   # shape: (block_h, block_w), values in 0 .. n_labels-1

    # Convert label indices to mask values in [0, 1]:
    # label 0      → mask 1.0 (block belongs to x1)
    # label n-1    → mask 0.0 (block belongs to x2)
    mask = 1.0 - labels.astype(np.float32) / (n_labels - 1)
    return mask   # (block_h, block_w)


# ---------------------------------------------------------------------------
# Transport plan
# ---------------------------------------------------------------------------

def _mask_transport(
    mask: Tensor,
    grad_pool: Tensor,
    cost_matrix: Tensor,
    eps: float = 0.8,
) -> Tensor:
    """Compute the optimal block transport plan for one source image.

    Given a binary-ish mask (which blocks are exposed for this source), find an
    assignment of *source* blocks to *target* positions that:
      - Sends high-saliency source blocks to exposed (mask=1) target positions
      - Minimises the total transport distance (using the cost matrix)

    This is a bipartite matching problem solved by an iterative auction
    algorithm (not Sinkhorn — the original Puzzle Mix uses this simpler method
    which converges fast for small grids).

    Parameters
    ----------
    mask : Tensor, shape (B, 1, block_h, block_w)
        The spatial mask (from graph cut).  Values in [0, 1].  For this
        function, we treat it as binary (> 0 = exposed position).
    grad_pool : Tensor, shape (B, block_h, block_w)
        Normalised saliency pooled to the block grid.  Used to score which
        source blocks are most valuable to transport to exposed positions.
    cost_matrix : Tensor, shape (1, n, n)
        Pairwise transport cost; n = block_h * block_w.
    eps : float
        Transport cost coefficient.  Larger eps makes distance more costly
        and keeps blocks closer to their original positions.

    Returns
    -------
    plan : Tensor, shape (B, n, n)
        Doubly-stochastic assignment matrix.  plan[b, i, j] = 1 means source
        block j is transported to target position i.
    """
    batch_size = mask.shape[0]
    n = mask.shape[-1] * mask.shape[-2]   # block_h * block_w

    # Number of auction iterations: same as the reference (= min grid dimension)
    block_h, block_w = mask.shape[-2], mask.shape[-1]
    n_iter = min(block_h, block_w)

    # z: binary indicator of which target positions are exposed by this mask
    z = (mask > 0).float()   # (B, 1, block_h, block_w)

    # Cost of transporting source block j to target position i:
    #   eps * distance(i, j)  -  grad_pool[i] * z[j]
    # The second term rewards sending high-saliency blocks to exposed positions.
    # grad_pool: (B, block_h, block_w) → (B, n, 1)
    # z:         (B, 1, block_h, block_w) → (B, 1, n)
    cost = (eps * cost_matrix
            - grad_pool.reshape(batch_size, n, 1) * z.reshape(batch_size, 1, n))
    # cost shape: (B, n, n)

    # Iterative auction: each source block (row) bids for the cheapest target
    # position (column).  Conflicts resolved by keeping only the lowest-cost
    # bidder per column; losers re-bid in the next iteration.
    for _ in range(n_iter):
        # Each source block picks the target with the lowest cost
        row_best = cost.min(-1)[1]              # (B, n)
        plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

        # Resolve conflicts: if multiple sources bid for the same target,
        # only the one with the lowest cost wins.
        cost_fight = plan * cost
        col_best = cost_fight.min(-2)[1]        # (B, n)
        plan_win = (torch.zeros_like(cost)
                    .scatter_(-2, col_best.unsqueeze(-2), 1) * plan)
        plan_lose = (1 - plan_win) * plan       # losing bids

        # Penalise losing bids so they seek other targets in the next round.
        # Adding 1 to the cost of the losing bid's chosen column effectively
        # makes that column less attractive for the next iteration.
        cost += plan_lose

    return plan_win   # (B, n, n)


# ---------------------------------------------------------------------------
# Apply transport plan to image blocks
# ---------------------------------------------------------------------------

def _transport_image(
    img: Tensor,
    plan: Tensor,
    batch_size: int,
    block_h: int,
    block_w: int,
    block_size: int,
) -> Tensor:
    """Rearrange image blocks according to a transport plan.

    Decomposes the image into (block_h × block_w) non-overlapping blocks of
    size (block_size × block_size), permutes the blocks according to `plan`,
    and reassembles the image.

    The region of the image processed is (block_h*block_size, block_w*block_size).
    Any columns/rows beyond this region (due to non-divisible W) are left
    unchanged in the caller.

    Parameters
    ----------
    img : Tensor, shape (B, C, H, W)
        Full input image.  Only the valid region is operated on.
    plan : Tensor, shape (B, n, n), where n = block_h * block_w
        plan[b, i, j] = 1 means block j is transported to target position i.
    batch_size, block_h, block_w, block_size : int
        Grid and block size parameters.

    Returns
    -------
    Tensor, shape (B, C, H, W)
        Image with blocks rearranged in the valid region; remainder unchanged.

    Decomposition walkthrough (for clarity)
    ----------------------------------------
    Let H' = block_h * block_size, W' = block_w * block_size, n = block_h * block_w.

    1. reshape (B,C,H',W') → (B,C,block_h,block_size,block_w*block_size)
       Split H' into (block_h groups) × (block_size pixels per group).
    2. .transpose(-2,-1) → (B,C,block_h, block_w*block_size, block_size)
       Swap the pixel-row dim and the combined-col dim (memory rearrangement).
    3. reshape → (B,C,block_h,block_w,block_size,block_size)
       Split block_w*block_size into (block_w groups) × (block_size pixels).
    4. .transpose(-2,-1) → (B,C,block_h,block_w,block_size,block_size)
       Swap pixel dims so inner 2D block is (row, col) order.
    5. reshape → (B,C,n,block_size,block_size)
       Flatten the (block_h,block_w) grid into n blocks.
    6. .permute(0,1,3,4,2).unsqueeze(-1) → (B,C,block_size,block_size,n,1)
    7. plan.T matmul: (B,1,1,1,n,n) × (B,C,block_size,block_size,n,1) → (B,C,block_size,block_size,n,1)
       Each of the n target positions receives exactly one source block.
    8. Reverse steps 6→1 to reassemble the spatial image.
    """
    C = img.shape[1]
    H_valid = block_h * block_size
    W_valid = block_w * block_size

    x = img[:, :, :H_valid, :W_valid].clone()   # work on valid region only

    # --- Decompose into blocks ---
    x = x.reshape(batch_size, C, block_h, block_size, block_w * block_size)
    x = x.transpose(-2, -1)
    x = x.reshape(batch_size, C, block_h, block_w, block_size, block_size)
    x = x.transpose(-2, -1)   # (B, C, block_h, block_w, block_size, block_size)
    x = x.reshape(batch_size, C, block_h * block_w, block_size, block_size)
    x = x.permute(0, 1, 3, 4, 2).unsqueeze(-1)   # (B, C, bs, bs, n, 1)

    # --- Apply transport plan ---
    transported = (
        plan.transpose(-2, -1)               # (B, n, n)
        .unsqueeze(1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 1, n, n)
        .matmul(x)                           # (B, C, bs, bs, n, 1)
        .squeeze(-1)
        .permute(0, 1, 4, 2, 3)             # (B, C, n, bs, bs)
    )

    # --- Reassemble ---
    transported = transported.reshape(batch_size, C, block_h, block_w, block_size, block_size)
    transported = transported.transpose(-2, -1)
    transported = transported.reshape(batch_size, C, block_h, block_w * block_size, block_size)
    transported = transported.transpose(-2, -1)
    transported = transported.reshape(batch_size, C, H_valid, W_valid)

    # Paste valid region back into a full-size copy of the original image
    result = img.clone()
    result[:, :, :H_valid, :W_valid] = transported
    return result


# ---------------------------------------------------------------------------
# Main Puzzle Mix function
# ---------------------------------------------------------------------------

def apply_puzzle_mix(
    x: Tensor,
    y_onehot: Tensor,
    saliency: Tensor,
    alpha: float = 0.4,
    beta: float = 1.2,
    gamma: float = 0.5,
    eta: float = 0.2,
    neigh_size: int = 4,
    n_labels: int = 3,
    use_transport: bool = True,
    t_eps: float = 0.8,
    block_size: int = 32,
    device: str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Apply Puzzle Mix to a batch of spectrograms.

    This is the main entry point called from the training loop.  The saliency
    map must be pre-computed by the training loop (in model.eval() mode) and
    passed in — apply_puzzle_mix does not access the model.

    Parameters
    ----------
    x : Tensor, shape (B, 1, H, W)
        Batch of spectrograms.
    y_onehot : Tensor, shape (B, num_classes)
        One-hot (or already soft) label vectors.
    saliency : Tensor, shape (B, H, W)
        Per-pixel saliency magnitude: |∂ℓ/∂x| computed in the training loop.
        Must be non-negative (absolute value should already be applied).
    alpha : float
        Beta distribution mixing coefficient.  Same parameter as Vanilla Mixup
        and CutMix (allows direct comparison of conditions at the same α).
    beta : float
        Pairwise smoothness weight.  Penalises graph-cut masks that cross
        regions of high local coherence.  Default 1.2 from Kim et al.
    gamma : float
        Data smoothness scaling.  Scales the local image gradient term relative
        to the label smoothness.  Default 0.5 from Kim et al.
    eta : float
        Label prior weight.  Penalises labels far from the overall mixing ratio α.
        Default 0.2 from Kim et al.
    neigh_size : int
        Resolution (in pixels) at which local smoothness is computed.  The
        input is pooled to (H // neigh_size, W // neigh_size) before computing
        pairwise penalties.  Default 4 from Kim et al.
    n_labels : int
        Number of discrete mixing levels for the graph cut.  3 → {0, 0.5, 1.0};
        2 → {0, 1} (binary).  Default 3 from Kim et al.
    use_transport : bool
        Whether to apply the optimal transport step.  When True, high-saliency
        blocks are moved into the mask-exposed regions before mixing.  This is
        the key step that makes Puzzle Mix more than just a guided CutMix.
    t_eps : float
        Transport cost coefficient.  Larger values penalise moving blocks far
        from their original position.  Default 0.8 from Kim et al.
    block_size : int
        Pixel size of each block (same in both directions — square blocks).
        Must be a divisor of H and should divide W with at most ~5% remainder.
        Sampled randomly from {16, 32} in the training loop.
    device : str
        Computation device.

    Returns
    -------
    x_mixed : Tensor, shape (B, 1, H, W)
        Mixed spectrograms.
    y_mixed : Tensor, shape (B, num_classes)
        Soft labels: ratio * y1 + (1 - ratio) * y2, where ratio is the mean
        mask value (fraction of the image covered by x1).

    Notes on block grid and valid region
    -------------------------------------
    For H=128, W=500, block_size=32:
      block_h = 128 // 32 = 4
      block_w = 500 // 32 = 15   (valid_W = 480, remainder = 20 px = 4%)
    The last 20 columns are not processed by the graph cut or transport; they
    receive the nearest-neighbour interpolation of the block-grid mask.

    Notes on beta normalisation
    ---------------------------
    Following the reference: beta = beta / max(block_h, block_w) / 16.
    This keeps the pairwise penalty at a consistent scale as the block grid
    changes.  The factor of 16 is empirical (from Kim et al. using 32-px
    images with 16-px minimum block size).
    """
    batch_size, C, H, W = x.shape
    assert C == 1, f"Expected single-channel spectrogram, got C={C}"

    # --- Derive block grid dimensions ---
    block_h = H // block_size
    block_w = W // block_size
    valid_W = block_w * block_size   # may be < W if W not divisible

    # Normalise beta (match reference convention)
    beta_norm = beta / max(block_h, block_w) / 16.0

    # --- Sample mixing coefficient and permutation ---
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    lam = max(lam, 1.0 - lam)

    perm = torch.randperm(batch_size, device=device)

    # --- Compute unary terms (saliency pooled to block grid) ---
    # Pool saliency from (B, H, W) → (B, block_h, block_w)
    # avg_pool2d requires 4D input; add a dummy channel dim temporarily
    saliency_4d = saliency.unsqueeze(1)   # (B, 1, H, W)
    grad1_pool = F.avg_pool2d(
        saliency_4d[:, :, :H, :valid_W],
        kernel_size=block_size,
        stride=block_size,
    ).squeeze(1)   # (B, block_h, block_w)

    # Normalise so each saliency map sums to 1 (probability distribution over blocks)
    # Add eps to avoid division by zero for zero-gradient inputs
    denom = grad1_pool.reshape(batch_size, -1).sum(1).reshape(batch_size, 1, 1) + 1e-8
    unary1_norm = grad1_pool / denom             # saliency of x1 (original order)
    unary2_norm = unary1_norm[perm]              # saliency of x2 (permuted order)

    # --- Compute pairwise smoothness terms ---
    # Pool input images to neigh_size resolution for local coherence computation
    k = block_size // neigh_size   # number of neigh_size-pixels per block side

    # Work on the valid region only
    x_valid = x[:, :, :H, :valid_W]
    input1_pool = F.avg_pool2d(x_valid, kernel_size=neigh_size, stride=neigh_size)
    input2_pool = input1_pool[perm]

    pw_x_t, pw_y_t = _neigh_penalty(input1_pool, input2_pool, k, k)
    pw_x_t2, pw_y_t2 = _neigh_penalty(input2_pool, input2_pool, k, k)
    pw_x_t3, pw_y_t3 = _neigh_penalty(input1_pool, input1_pool, k, k)

    # Pairwise costs: scale by beta_norm * gamma
    # Following reference: uses 4 combinations of (input1, input2) smoothness
    # to account for cuts where either source or the blend crosses a boundary.
    # Here we simplify to the symmetric average for the rectangular case.
    pw_x = beta_norm * gamma * (pw_x_t + pw_x_t2 + pw_x_t3) / 3.0
    pw_y = beta_norm * gamma * (pw_y_t + pw_y_t2 + pw_y_t3) / 3.0

    # --- Solve graph cut for each example ---
    # Convert to numpy for pygco; detach from computation graph
    unary1_np = unary1_norm.detach().cpu().numpy()   # x1 saliency
    unary2_np = unary2_norm.detach().cpu().numpy()   # x2 saliency
    pw_x_np = pw_x.detach().cpu().numpy()
    pw_y_np = pw_y.detach().cpu().numpy()

    masks = []
    for i in range(batch_size):
        # NOTE: argument order is (x2_saliency, x1_saliency) to match the
        # reference convention — see _graphcut_single docstring for explanation.
        mask_i = _graphcut_single(
            x2_unary=unary2_np[i],
            x1_unary=unary1_np[i],
            pw_x=pw_x_np[i],
            pw_y=pw_y_np[i],
            alpha=lam,
            beta=beta_norm,
            eta=eta,
            n_labels=n_labels,
        )
        masks.append(mask_i)

    # Stack block-resolution masks: (B, block_h, block_w)
    mask_block = torch.tensor(
        np.stack(masks, axis=0), dtype=torch.float32, device=device,
    )
    mask_block = mask_block.unsqueeze(1)   # (B, 1, block_h, block_w)

    # --- Optional transport step ---
    if use_transport:
        cost_mat = compute_cost_matrix(block_h, block_w, device)

        # Transport x1: move high-saliency x1 blocks into mask=1 (x1) positions
        plan1 = _mask_transport(mask_block, unary1_norm, cost_mat, eps=t_eps)
        x1_transported = _transport_image(x, plan1, batch_size, block_h, block_w, block_size)

        # Transport x2: move high-saliency x2 blocks into mask=0 (x2) positions
        plan2 = _mask_transport(1.0 - mask_block, unary2_norm, cost_mat, eps=t_eps)
        x2_transported = _transport_image(x[perm], plan2, batch_size, block_h, block_w, block_size)
    else:
        x1_transported = x
        x2_transported = x[perm]

    # --- Upsample block mask to pixel resolution ---
    # nearest-neighbour interpolation: block boundary artefacts are avoided
    # because the mask is piecewise constant at block granularity anyway.
    mask_pixel = F.interpolate(
        mask_block,
        size=(H, W),
        mode='nearest',
    )   # (B, 1, H, W)

    # --- Blend ---
    x_mixed = mask_pixel * x1_transported + (1.0 - mask_pixel) * x2_transported

    # --- Mix labels proportionally to actual mask coverage ---
    ratio = mask_pixel.reshape(batch_size, -1).mean(dim=-1)   # (B,)
    y_mixed = (ratio.unsqueeze(-1) * y_onehot
               + (1.0 - ratio).unsqueeze(-1) * y_onehot[perm])

    return x_mixed, y_mixed
