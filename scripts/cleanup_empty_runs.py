#!/usr/bin/env python3
import os
import shutil
import sys


def count_files_in_directory(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    try:
        entries = os.listdir(path)
    except Exception:
        return 0
    count = 0
    for name in entries:
        full = os.path.join(path, name)
        if os.path.isfile(full):
            count += 1
    return count

min_checkpoints = 6
def main():
    # Base directory to scan; default to outputs/exp relative to repo root
    base_dir = "/workspace/gan_seg/outputs/exp"

    if not os.path.isdir(base_dir):
        print(f"Base directory not found: {base_dir}")
        return 0

    removed = []
    kept = []

    for item in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, item)
        if not os.path.isdir(run_dir):
            continue
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        num_files = count_files_in_directory(checkpoints_dir)
        if num_files < min_checkpoints:
            print(f"Removing: {run_dir} (checkpoints files: {num_files})")
            
            try:
                shutil.rmtree(run_dir)
                removed.append((run_dir, num_files))
            except Exception as e:
                print(f"Failed to remove {run_dir}: {e}")
        else:
            kept.append((run_dir, num_files))

    # for path, n in removed:
    #     print(f"Removed: {path} (checkpoints files: {n})")
    # for path, n in kept:
    #     print(f"Kept:    {path} (checkpoints files: {n})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


