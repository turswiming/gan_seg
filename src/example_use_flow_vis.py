"""
使用示例：可视化 AV2 数据集的 scene flow
"""

from pathlib import Path
from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo
from visualize_flow_bev import (
    visualize_flow_bev,
    visualize_flow_bev_advanced,
    visualize_flow_bev_simple
)


def main():
    # 加载数据集
    av2_zoo = AV2SceneFlowZoo(
        root_dir=Path("/workspace/av2data/train"),
        expected_camera_shape=(1280, 720),
        eval_args={},
        with_rgb=False,
        flow_data_path=Path("/workspace/av2flow/train"),
        range_crop_type="ego",
        point_size=8192,
        load_flow=True,
        load_boxes=False,
    )
    
    # 获取一个样本
    item = av2_zoo[0]
    
    pc1 = item["point_cloud_first"]
    pc2 = item["point_cloud_next"]
    flow = item["flow"]
    sequence_id = item["sequence_id"]
    
    print(f"Sequence ID: {sequence_id}")
    print(f"PC1 shape: {pc1.shape}")
    print(f"PC2 shape: {pc2.shape}")
    print(f"Flow shape: {flow.shape}")
    
    # 方法 1：基础可视化
    visualize_flow_bev(
        pc1, pc2, flow,
        save_path=f"flow_bev_{sequence_id[:8]}.png",
        title=f"Scene Flow - {sequence_id[:8]}",
        subsample=2000
    )
    
    # 方法 2：高级可视化（4 子图）
    visualize_flow_bev_advanced(
        pc1, pc2, flow,
        save_path=f"flow_advanced_{sequence_id[:8]}.png",
        subsample=3000
    )
    
    # 方法 3：快速可视化
    visualize_flow_bev_simple(
        pc1, pc2, flow,
        save_path=f"flow_simple_{sequence_id[:8]}.png"
    )
    
    print("\n可视化完成！")


if __name__ == "__main__":
    main()

