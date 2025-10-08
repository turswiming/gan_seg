import os
import sys
import torch
import torch.nn.functional as F

# Ensure project root is in path (mimic flow_smooth_loss_eval structure)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from losses.ReconstructionLoss import ReconstructionLoss
from losses.ReconstructionLoss_optimized import ReconstructionLossOptimized


def generate_dummy_batch(batch_size=2, num_points=4096, num_slots=4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    pc1_list = []
    pc2_list = []
    mask_list = []
    flow_list = []

    for _ in range(batch_size):
        pc1 = torch.randn(num_points, 3, device=device)
        flow = 0.01 * torch.randn(num_points, 3, device=device)
        pc2 = pc1 + flow

        # Random logits for masks, softmax applied inside loss
        mask_logits = torch.randn(num_slots, num_points, device=device)

        pc1_list.append(pc1)
        pc2_list.append(pc2)
        mask_list.append(mask_logits)
        flow_list.append(flow)

    return pc1_list, pc2_list, mask_list, flow_list


def tensor_list_max_abs_diff(list_a, list_b):
    if len(list_a) != len(list_b):
        return float("inf")
    diffs = []
    for a, b in zip(list_a, list_b):
        if a.shape != b.shape:
            return float("inf")
        diffs.append((a - b).abs().max().item())
    return max(diffs) if diffs else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate both losses
    loss_ref = ReconstructionLoss(device)
    loss_opt = ReconstructionLossOptimized(device, use_checkpointing=False, chunk_size=2048)

    # Deterministic dummy data
    pc1_list, pc2_list, mask_list, flow_list = generate_dummy_batch(
        batch_size=2, num_points=2048, num_slots=3, device=device
    )

    # Run both implementations
    loss_ref_val, rec_ref = loss_ref(pc1_list, pc2_list, mask_list, flow_list)
    loss_opt_val, rec_opt = loss_opt(pc1_list, pc2_list, mask_list, flow_list)

    # Compare results
    loss_diff = (loss_ref_val - loss_opt_val).abs().item()
    rec_diff = tensor_list_max_abs_diff(rec_ref, rec_opt)

    # Note: The reference implementation appears to return the last batch loss
    # whereas the optimized version returns batch-averaged loss.
    # We report raw differences without modifying either implementation.

    print(f"device: {device}")
    print(f"loss_ref: {loss_ref_val.item():.8f}")
    print(f"loss_opt: {loss_opt_val.item():.8f}")
    print(f"|loss_ref - loss_opt|: {loss_diff:.8e}")
    print(f"max_abs_diff(reconstructed point clouds): {rec_diff:.8e}")

    # Simple verdict based on tolerances
    tol = 1e-5
    same_loss = loss_diff < tol
    same_rec = rec_diff < tol
    print(f"consistent_loss_within_{tol}: {same_loss}")
    print(f"consistent_reconstruction_within_{tol}: {same_rec}")


if __name__ == "__main__":
    main()



