from pathlib import Path
import argparse

from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_scene_flow import (
    ArgoverseSceneFlowSequenceLoader,
)


def main():
    parser = argparse.ArgumentParser(
        description="Test ArgoverseSceneFlowSequenceLoader"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to AV2 raw sequence root (contains per-sequence subfolders)",
    )
    parser.add_argument(
        "--flow_dir",
        type=str,
        default=None,
        help="Path to flow feather root matching raw_dir (optional; defaults to *_sceneflow_feather)",
    )
    parser.add_argument(
        "--use_gt_flow",
        action="store_true",
        help="Use ground-truth scene flow (default False if not set)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="How many sequence IDs to preview",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    flow_path = Path(args.flow_dir) if args.flow_dir is not None else None

    loader = ArgoverseSceneFlowSequenceLoader(
        raw_data_path=raw_path,
        flow_data_path=flow_path,
        use_gt_flow=args.use_gt_flow,
    )

    seq_ids = loader.get_sequence_ids()
    print(f"Total sequences: {len(seq_ids)}")
    print("First IDs:", seq_ids[: args.limit])

    # Load first sequence and peek at a sample pair
    if len(seq_ids) == 0:
        print("No sequences found.")
        return

    seq = loader[0]
    print(f"Sequence '{seq_ids[0]}' length (num frames): {len(seq)}")

    if len(seq) >= 2:
        frame0, raw0 = seq.load(0, 1, with_flow=True)
        # frame0 is expected to be a TimeSyncedSceneFlowFrame
        pc0 = frame0.pc_0.points
        pc1 = frame0.pc_1.points
        flow = frame0.flow.vecs if hasattr(frame0.flow, "vecs") else None
        print(
            f"pc0: {pc0.shape}, pc1: {pc1.shape}, flow: {None if flow is None else flow.shape}"
        )
        print(
            f"pc0 min/max: ({pc0.min():.3f},{pc0.max():.3f}), pc1 min/max: ({pc1.min():.3f},{pc1.max():.3f})"
        )
    else:
        print("Sequence has fewer than 2 frames; skipping sample load.")


if __name__ == "__main__":
    main()



