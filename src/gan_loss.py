import torch
from torch import nn
from torch.nn import functional as F
import open3d as o3d

# This code is originally from OGC
def fit_motion_svd_batch(pc1, pc2, pc1_mask, pc2_mask=None):
    """
    :param pc1: (B, N, 3) torch.Tensor.
    :param pc2: (B, N, 3) torch.Tensor.
    :param mask: (B, N) torch.Tensor.
    :return:
        R_base: (B, 3, 3) torch.Tensor.
        t_base: (B, 3) torch.Tensor.
    """
    n_batch, n_point, _ = pc1.size()
    pc1_mask = pc1_mask.type_as(pc1)  # Ensure pc1_mask has the same type as pc1
    pc2_mask = pc2_mask.type_as(pc2)  # Ensure pc2_mask has the same type as pc2
    pc1_mask = pc1_mask.unsqueeze(0)  # (B, N)
    pc2_mask = pc2_mask.unsqueeze(0)  # (B, N)
    pc1_mean = torch.einsum('bnd,bn->bd', pc1, pc1_mask) / torch.sum(pc1_mask, dim=1, keepdim=True)   # (B, 3)
    pc1_mean.unsqueeze_(1)
    pc2_mean = torch.einsum('bnd,bn->bd', pc2, pc2_mask) / torch.sum(pc2_mask, dim=1, keepdim=True)
    pc2_mean.unsqueeze_(1)

    pc1_centered = pc1 - pc1_mean
    pc2_centered = pc2 - pc2_mean

    pc1_masked = torch.diag_embed(pc1_mask).bmm(pc1_centered)
    pc2_masked = torch.diag_embed(pc2_mask).bmm(pc2_centered)
    S = pc1_masked.transpose(1, 2).bmm(pc2_masked)
    # We just return an identity matrix.
    valid_batches = ~torch.isnan(S).any(dim=1).any(dim=1)
    R_base = torch.eye(3, device=pc1.device).unsqueeze(0).repeat(n_batch, 1, 1)
    t_base = torch.zeros((n_batch, 3), device=pc1.device)

    if valid_batches.any():
        S = S[valid_batches, ...]
        u, s, v = torch.svd(S, some=False, compute_uv=True)
        R = torch.bmm(v, u.transpose(1, 2))
        det = torch.det(R)

        # Correct reflection matrix to rotation matrix
        diag = torch.ones_like(S[..., 0], requires_grad=False)
        diag[:, 2] = det
        R = v.bmm(torch.diag_embed(diag).bmm(u.transpose(1, 2)))

        pc1_mean, pc2_mean = pc1_mean[valid_batches], pc2_mean[valid_batches]
        t = pc2_mean.squeeze(1) - torch.bmm(R, pc1_mean.transpose(1, 2)).squeeze(2)

        R_base[valid_batches] = R.to(R_base.dtype)
        t_base[valid_batches] = t.to(t_base.dtype)

    return R_base, t_base


class GanLoss():
    def __init__(self, device):
        self.device = device
        self.criterion = nn.MSELoss()
        self.KNN_SEARCH_SIZE = 3
        


    def pi_func(self, mask_single_frame, point_position, sample_goal):
        """
        使用 Open3D 的 KDTree 实现 KNN 搜索，并手动计算梯度。
        :param mask_single_frame: (N,)
        :param point_position: (N, 3)
        :param sample_goal: (num_tracks, 3)
        :return: (num_tracks,)
        """
        # 确保输入形状正确
        if len(point_position.shape) == 3:
            point_position = point_position.squeeze(0)  # 去掉批次维度
        if len(sample_goal.shape) == 3:
            sample_goal = sample_goal.squeeze(0)  # 去掉批次维度

        num_tracks = sample_goal.shape[0]
        N = point_position.shape[0]

        # 使用 Open3D 构建 KDTree
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_position.cpu().detach().numpy())  # 转换为 numpy
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        # 初始化结果
        neighbor_indices = []
        for i in range(num_tracks):
            query = sample_goal[i].cpu().numpy()  # 转换为 numpy
            _, idx, _ = kdtree.search_knn_vector_3d(query, self.KNN_SEARCH_SIZE)
            neighbor_indices.append(idx)

        # 转换为 PyTorch 张量
        neighbor_indices = torch.tensor(neighbor_indices, device=self.device)  # (num_tracks, k)

        # 聚合 mask_single_frame 的值
        neighbor_mask_values = mask_single_frame[neighbor_indices]  # (num_tracks, k)

        # 使用 softmax 权重计算加权平均
        diff = point_position[neighbor_indices] - sample_goal.unsqueeze(1)  # (num_tracks, k, 3)
        dist = torch.norm(diff, dim=2)  # (num_tracks, k)
        temperature = 0.05  # 可调节
        weights = torch.softmax(-dist / temperature, dim=1)  # (num_tracks, k)

        # 聚合 mask 值
        neighbor_values = torch.sum(weights * neighbor_mask_values, dim=1)  # (num_tracks,)

        return neighbor_values

    def __call__(self, inputs,pred_mask, pred_flow):
        point_cloud_first = inputs["point_cloud_first"].to(self.device)
        point_cloud_second = inputs["point_cloud_second"].to(self.device)
        pred_mask = pred_mask.to(self.device)
        softmaxed_pred_mask = F.softmax(pred_mask, dim=0)
        B, N = pred_mask.shape
        for b in range(B):
            pred_mask_b = softmaxed_pred_mask[b]
            half_mask = torch.zeros_like(pred_mask_b,device=self.device)
            half_mask[::2] = 1
            mask_next_half1 = self.pi_func(pred_mask_b*half_mask, point_cloud_first+pred_flow, point_cloud_second)
            mask_next_half1 = mask_next_half1.view(-1)
            rot1, move1 = fit_motion_svd_batch(
                point_cloud_first,
                point_cloud_second,
                (pred_mask_b*half_mask),
                mask_next_half1
            )
            half_mask2 = torch.zeros_like(pred_mask_b,device=self.device)
            half_mask2[1::2] = 1
            mask_next_half2 = self.pi_func(pred_mask_b*half_mask2, point_cloud_first+pred_flow, point_cloud_second)
            mask_next_half2 = mask_next_half2.view(-1)
            rot2, move2 = fit_motion_svd_batch(
                point_cloud_first,
                point_cloud_second,
                (pred_mask_b*half_mask2),
                mask_next_half2
            )
            loss = torch.zeros(1, device=self.device)
            loss += self.criterion(rot1, rot2)
            loss += self.criterion(move1, move2)
        #another loss
        flow = point_cloud_second - point_cloud_first
        pred_flow = pred_flow.view(-1, 3)
        loss_flow = self.criterion(pred_flow, flow)
        print("loss_flow", loss_flow.item())
        return loss