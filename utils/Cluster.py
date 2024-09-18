import torch
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import json
import cv2
import os
from tqdm import tqdm, trange
from scipy import stats

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def find_row_indices(a, b):
    indices = []
    for row in b:
        comparison = a == row
        index = np.where(comparison.all(axis=1))[0]
        indices.append(index)

    return np.array(indices).reshape(-1, 1)


def find_nearest(data, centers):
    nearest_points = []
    for center in centers:
        distances = np.linalg.norm(data - center, axis=1)
        nearest_point = data[np.argmin(distances)]
        nearest_points.append(nearest_point)
    return np.array(nearest_points)


# ! 聚类相关的内容
class Cluster_Manager:
    def __init__(self, class_num=0, cluster_config_file=None):
        self.class_num = class_num
        self.clusters = []
        self.position = []
        if cluster_config_file is not None:
            self.load(cluster_config_file)
        return

    def load(self, cluster_config_file):
        with open(os.path.join(cluster_config_file, 'clusters.json'), 'r') as f:
            data = json.load(f)
        self.class_num = data['class_num']
        configs = data['cluster_dirs']
        assert self.class_num == len(configs)
        self.clusters = []
        for i, config in enumerate(configs):
            if config is None:
                cluster = None
            else:
                dir = os.path.join(cluster_config_file, 'c' + str(i))
                cluster = Cluster(cluster_dir=dir)
            self.clusters.append(cluster)

    def save(self, cluster_manager_dir):
        os.makedirs(cluster_manager_dir, exist_ok=True)
        cluster_dirs = []
        cluster_manager_config_path = os.path.join(cluster_manager_dir, 'clusters.json')
        for i, cluster in enumerate(self.clusters):
            if cluster is None:
                cluster_dirs.append(None)
            else:
                cluster_dir = os.path.join(cluster_manager_dir, 'c' + str(i))
                cluster.save(cluster_dir)
                cluster_dirs.append(cluster_dir)
        manager_data = {'class_num': self.class_num, 'cluster_dirs': cluster_dirs}
        with open(cluster_manager_config_path, "w") as f:
            json.dump(manager_data, f)
            print('successfully save cluster manager to:', cluster_manager_config_path)

    # ! 根据一组点以及他们的标签，对每一类点进行分别聚类，构造对应的Cluster类
    def update_center(self, labels, pixels, quantile=0.3, n_samples=5000, band_factor=0.5):
        print("updating clusers...")
        self.clusters = []
        # self.position = []
        # if self.class_num==1:
        #     cluster = Cluster()
        #     cluster.update_center(pixels.reshape(-1,3), quantile=quantile, n_samples=n_samples, band_factor = band_factor)
        #     self.clusters.append(cluster)
        #     return
        for i in tqdm(range(self.class_num)):
            class_idx = np.squeeze(labels == i)  # the specific class index
            # self.position_idx = position[class_idx]
            # class_roughness = roughness[class_idx]
            # class_metallic = metallic[class_idx]
            class_pixels = pixels[class_idx]  # the specific class pixels
            if len(class_pixels) == 0:
                self.clusters.append(None)
                print('no pixel belongs to class:', i)
                continue
            cluster = Cluster()
            cluster.update_center(class_pixels, quantile=quantile, n_samples=n_samples, band_factor=band_factor)
            self.clusters.append(cluster)
            # self.position.append(self.position_idx[index])
        return

    # ! 根据一组点的标签，将这组点映射到聚类后的颜色
    def dest_color(self, rgb, label):
        result = rgb.clone()
        # roughness = torch.zeros((rgb.shape[0],),dtype=torch.float32).to(rgb.device)
        # metallic = torch.zeros((rgb.shape[0],),dtype=torch.float32).to(rgb.device)
        if self.class_num == 1:
            result = self.clusters[0].dest_color(rgb)
            return result
        for i in range(self.class_num):
            if self.clusters[i] is None:
                continue
            class_idx = torch.squeeze(label == i)
            class_rgb = rgb[class_idx]
            if class_rgb.shape[0] == 0:
                continue
            result[class_idx] = (self.clusters[i].dest_color(class_rgb)).reshape(-1, 1)
        return result

    def dest_class(self, rgb, label):
        result = torch.zeros([rgb.shape[0], 1], dtype=torch.long).to(rgb.device)

        for i in range(self.class_num):
            if self.clusters[i] is None:
                continue
            class_idx = torch.squeeze(label == i)
            class_rgb = rgb[class_idx]
            if class_rgb.shape[0] == 0:
                continue
            result[class_idx] = (self.clusters[i].dest_class(class_rgb)).float()
        return result


class Cluster:
    def __init__(self, device=torch.device('cuda'), intensity_factor=0.5, cluster_dir=None):
        self.batch_size = 10240  #
        self.anchors = None  #
        self.links = None  #
        self.rgb_centers = None
        self.device = device  #
        self.intensity_factor = intensity_factor  #
        if cluster_dir is not None:
            self.load(cluster_dir)

    def load(self, cluster_dir):
        with open(os.path.join(cluster_dir, 'config.json'), 'r') as f:
            data = json.load(f)
        self.batch_size = data['batch_size']
        self.intensity_factor = data['intensity_factor']
        self.anchors = torch.Tensor(data['anchors']).to(self.device)
        self.rgb_centers = torch.Tensor(data['rgb_centers']).to(self.device)
        self.links = torch.Tensor(data['links']).long().to(self.device)
        return

    def save(self, cluster_dir):
        os.makedirs(cluster_dir, exist_ok=True)
        cluster_data = {"batch_size": self.batch_size, "intensity_factor": self.intensity_factor,
                        "rgb_centers": self.rgb_centers.cpu().numpy().tolist(), \
                        "anchors": self.anchors.cpu().numpy().tolist(), "links": self.links.cpu().numpy().tolist()}
        cluster_config_path = os.path.join(cluster_dir, 'config.json')
        with open(cluster_config_path, "w") as f:
            json.dump(cluster_data, f)
            print('successfully save cluster to:', cluster_config_path)

        for i in range(self.rgb_centers.shape[0]):
            color = self.rgb_centers[i]
            color_img = to8b(np.ones((50, 50, 3)) * color.cpu().numpy())
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(cluster_dir, str(i) + '.png'), color_img)
        return

    def update_center(self, pixels, quantile=0.3, n_samples=5000, band_factor=0.5):  # 对新采样的一组点进行聚类，更新类中心点
        # pixels = self.mapping_color_np(pixels)
        bandwidth = estimate_bandwidth(pixels, quantile=quantile, n_samples=n_samples)
        bandwidth = max(bandwidth * band_factor, 0.1)
        # print('bandwidth:',bandwidth)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(pixels)
        labels = ms.labels_
        labels_unique = np.unique(labels)
        # center = []
        # for label in labels_unique:
        #     center_idx = np.where(labels == label)[0]
        #     pixel = pixels[center_idx].squeeze()
        #     cen = np.median(pixel, axis=0)
        #     center.append(cen)
        n_clusters = len(labels_unique)
        # print("number of estimated clusters : %d" % n_clusters)
        self.choose_anchors(pixels, labels)
        centers = torch.from_numpy(ms.cluster_centers_).to(self.device)
        # centers = find_nearest(pixels,centers.cpu().numpy())
        # index = find_row_indices(pixels, centers)
        # self.roughness_center = class_roughness[index.flatten()]
        # self.metallic_center = class_metallic[index.flatten()]
        # centers = torch.from_numpy(centers).to(self.device)
        # self.rgb_centers = self.inv_mapping_color(centers)
        self.rgb_centers = centers

    # ! 对空间中的颜色点进行体素滤波，滤波筛选后的点作为锚点，保存每个锚点所属的颜色类别
    # ! 转换一个颜色的时候，就是将找到该颜色在颜色空间中最近的锚点，将这个锚点所属的类视为该颜色的类
    def choose_anchors(self, pixels, labels):
        pixels = torch.from_numpy(pixels).to(self.device).float()
        labels = torch.from_numpy(labels).to(self.device).long()
        print("before merge:", pixels.shape)
        print("choosing anchors...")
        N = pixels.shape[0]
        leaf_size = 0.001
        size_x = int(1 / leaf_size)
        # size_y = int(1/leaf_size)
        # size_z = int(1/leaf_size)
        voxel = torch.zeros((size_x, 1), dtype=torch.float32).to(self.device)
        voxel_label = torch.zeros((size_x, 1), dtype=torch.long).to(self.device) - 1
        half_leaf_size = leaf_size / 2

        id = torch.clamp((pixels / leaf_size).long(), 0, size_x - 1)
        voxel_center = id * leaf_size + half_leaf_size
        dist = torch.sum((voxel_center - pixels) ** 2, dim=1)
        _, sorted_indices = torch.sort(dist, descending=True)
        id = id[sorted_indices, :]
        pixels = pixels[sorted_indices, :]
        labels = labels[sorted_indices, None]
        voxel[id[:, 0]] = pixels
        voxel_label[id[:, 0]] = labels
        valid_voxel_id = torch.squeeze(voxel_label >= 0)
        self.anchors = voxel[valid_voxel_id]
        self.links = voxel_label[valid_voxel_id]
        print("after merge:", self.anchors.shape)

    def choose_anchors_cpu(self, pixels, labels):
        pixels = torch.from_numpy(pixels).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)
        print("before merge:", pixels.shape)
        print("choosing anchors...")
        N = pixels.shape[0]
        leaf_size = 0.01
        size_x = int(1 / leaf_size)
        size_y = int(1 / leaf_size)
        size_z = int(1 / leaf_size)
        voxel = torch.zeros(shape=(size_x, size_y, size_z, 3), dtype=torch.float32)
        voxel_near = torch.ones(shape=(size_x, size_y, size_z, 1), dtype=torch.float32)
        voxel_label = torch.zeros(shape=(size_x, size_y, size_z, 1), dtype=torch.int32) - 1
        half_leaf_size = leaf_size / 2

        id_x = torch.clamp(torch.int(pixels[:, 0] / leaf_size), 0, size_x - 1)
        id_y = torch.clamp(torch.int(pixels[:, 1] / leaf_size), 0, size_y - 1)
        id_z = torch.clamp(torch.int(pixels[:, 2] / leaf_size), 0, size_z - 1)
        pixel_v_center = [id_x * leaf_size + half_leaf_size, id_y * leaf_size + half_leaf_size,
                          id_z * leaf_size + half_leaf_size]

        id = torch.clamp(torch.int(pixels / leaf_size), 0, size_x - 1)
        voxel_center = id * leaf_size + half_leaf_size
        dist = self.compute_dist(voxel_center, pixels)
        sorted_logits, sorted_indices = torch.sort(dist, descending=True)
        print(pixels.shape, dist.shape, sorted_indices.shape)

        for i in range(N):
            id_x = np.clip(np.int(pixels[i, 0] / leaf_size), 0, size_x - 1)
            id_y = np.clip(np.int(pixels[i, 1] / leaf_size), 0, size_y - 1)
            id_z = np.clip(np.int(pixels[i, 2] / leaf_size), 0, size_z - 1)
            voxel_center = np.array([id_x * leaf_size + half_leaf_size, id_y * leaf_size + half_leaf_size,
                                     id_z * leaf_size + half_leaf_size])
            dist = np.linalg.norm(voxel_center - pixels[i])
            if (dist < voxel_near[id_x, id_y, id_z]):
                voxel_near[id_x, id_y, id_z] = dist
                voxel[id_x, id_y, id_z] = pixels[i]
                voxel_label[id_x, id_y, id_z] = labels[i]

        voxel = voxel.reshape(-1, 3)
        voxel_label = voxel_label.reshape(-1, 1)
        self.anchors = []
        self.links = []
        for i in range(voxel.shape[0]):
            if voxel_label[i] == -1:
                continue
            self.anchors.append(voxel[i, None])
            self.links.append(voxel_label[i, None])
        self.anchors = np.concatenate(self.anchors, 0)
        self.links = np.concatenate(self.links, 0)
        print("after merge:", self.anchors.shape)
        self.anchors = torch.from_numpy(self.anchors).to(self.device)
        self.links = torch.from_numpy(self.links).long().to(self.device)

    def choose_anchors0(self, pixels, labels):
        print("before merge:", pixels.shape)
        print("choosing anchors...")
        N = pixels.shape[0]
        leaf_size = 0.01
        size_x = np.int(1 / leaf_size)
        size_y = np.int(1 / leaf_size)
        size_z = np.int(1 / leaf_size)
        voxel = np.zeros(shape=(size_x, size_y, size_z, 3), dtype=np.float32)
        voxel_near = np.zeros(shape=(size_x, size_y, size_z, 1), dtype=np.float32) + 1
        voxel_label = np.zeros(shape=(size_x, size_y, size_z, 1), dtype=np.int32) - 1
        half_leaf_size = leaf_size / 2
        for i in range(N):
            id_x = np.clip(np.int(pixels[i, 0] / leaf_size), 0, size_x - 1)
            id_y = np.clip(np.int(pixels[i, 1] / leaf_size), 0, size_y - 1)
            id_z = np.clip(np.int(pixels[i, 2] / leaf_size), 0, size_z - 1)
            voxel_center = np.array([id_x * leaf_size + half_leaf_size, id_y * leaf_size + half_leaf_size,
                                     id_z * leaf_size + half_leaf_size])
            dist = np.linalg.norm(voxel_center - pixels[i])
            if (dist < voxel_near[id_x, id_y, id_z]):
                voxel_near[id_x, id_y, id_z] = dist
                voxel[id_x, id_y, id_z] = pixels[i]
                voxel_label[id_x, id_y, id_z] = labels[i]

        voxel = voxel.reshape(-1, 3)
        voxel_label = voxel_label.reshape(-1, 1)
        self.anchors = []
        self.links = []
        for i in range(voxel.shape[0]):
            if voxel_label[i] == -1:
                continue
            self.anchors.append(voxel[i, None])
            self.links.append(voxel_label[i, None])
        self.anchors = np.concatenate(self.anchors, 0)
        self.links = np.concatenate(self.links, 0)
        print("after merge:", self.anchors.shape)
        self.anchors = torch.from_numpy(self.anchors).to(self.device)
        self.links = torch.from_numpy(self.links).long().to(self.device)

    def dest_color(self, rgb):
        # d_rgb = self.mapping_color(rgb)
        d_rgb = rgb
        start_idx = 0
        idxs = []
        while start_idx < d_rgb.shape[0]:
            end_idx = min(d_rgb.shape[0], start_idx + self.batch_size)
            idx = self.nearest_anchor(d_rgb[start_idx:end_idx])
            idxs.append(idx)
            start_idx = end_idx
        idxs = torch.cat(idxs, 0)
        # self.roughness_center = (torch.from_numpy(self.roughness_center).to(self.device))
        # self.metallic_center = (torch.from_numpy(self.metallic_center).to(self.device))
        return torch.squeeze((self.rgb_centers[self.links[idxs]]).float())

    def dest_class(self, rgb):
        d_rgb = self.mapping_color(rgb)
        start_idx = 0
        idxs = []
        while start_idx < d_rgb.shape[0]:
            end_idx = min(d_rgb.shape[0], start_idx + self.batch_size)
            idx = self.nearest_anchor(d_rgb[start_idx:end_idx])
            idxs.append(idx)
            start_idx = end_idx
        idxs = torch.cat(idxs, 0)
        return self.links[idxs]

    def compute_dist(self, a, b):
        sq_a = a ** 2
        sum_sq_a = sq_a.unsqueeze(1)  # m->[m, 1]
        sq_b = (b ** 2).reshape(1, -1)
        sum_sq_b = sq_b.unsqueeze(0)  # n->[1, n]
        bt = b.t()
        return sq_a + sq_b - 2 * torch.mm(a, b.T)

    def nearest_anchor(self, d_rgb):
        dist = self.compute_dist(self.anchors, d_rgb)
        idx = torch.argmin(dist, dim=0).long()
        return idx

    def mapping_color0(self, rgb):
        return rgb

    def mapping_color_np(self, rgb):
        intensity = np.sum(rgb, axis=-1)
        d_rgb = np.zeros_like(rgb)
        d_rgb[..., 0] = intensity / 3.0 * self.intensity_factor
        d_rgb[..., 1] = rgb[..., 1] / intensity
        d_rgb[..., 2] = rgb[..., 2] / intensity
        return d_rgb

    def mapping_color(self, rgb):
        intensity = torch.sum(rgb, axis=-1)
        d_rgb = torch.zeros_like(rgb).to(self.device)
        d_rgb[..., 0] = intensity / 3.0 * self.intensity_factor
        d_rgb[..., 1] = rgb[..., 1] / intensity
        d_rgb[..., 2] = rgb[..., 2] / intensity
        return d_rgb

    def inv_mapping_color0(self, d_rgb):
        return d_rgb

    def inv_mapping_color(self, d_rgb):
        intensity = d_rgb[..., 0] * 3.0 / self.intensity_factor
        rgb = torch.zeros_like(d_rgb).to(self.device)
        rgb[..., 1] = d_rgb[..., 1] * intensity
        rgb[..., 2] = d_rgb[..., 2] * intensity
        rgb[..., 0] = intensity - rgb[..., 1] - rgb[..., 2]
        return rgb

