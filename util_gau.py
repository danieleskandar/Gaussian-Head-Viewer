import numpy as np
from plyfile import PlyData
from dataclasses import dataclass

def slice_data(start, end, data):
        return np.copy(data[0][start:end, :]), np.copy(data[1][start:end, :]), np.copy(data[2][start:end, :]), np.copy(data[3][start:end, :]), np.copy(data[4][start:end, :])

@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray

    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-1]

    def stats(self):
        self.xyz_mean = np.mean(self.xyz, axis=0)
        self.xyz_var = np.var(self.xyz, axis=0)
        self.scale_mean = np.mean(self.scale, axis=0)
        self.scale_var = np.var(self.scale, axis=0)

    def get_data(self):
        return np.copy(self.xyz), np.copy(self.rot), np.copy(self.scale), np.copy(self.opacity), np.copy(self.sh)

def naive_gaussian():
    gau_xyz = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)

    gau_rot = np.array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0
    ]).astype(np.float32).reshape(-1, 4)

    gau_s = np.array([
        0.03, 0.03, 0.03,
        0.2, 0.03, 0.03,
        0.03, 0.2, 0.03,
        0.03, 0.03, 0.2
    ]).astype(np.float32).reshape(-1, 3)

    gau_c = np.array([
        1, 0, 1, 
        1, 0, 0, 
        0, 1, 0, 
        0, 0, 1, 
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = (gau_c - 0.5) / 0.28209
    
    gau_a = np.array([
        1, 1, 1, 1
    ]).astype(np.float32).reshape(-1, 1)

    return GaussianData( gau_xyz, gau_rot, gau_s, gau_a, gau_c)

def random_gaussian(gaussians):
    sample_ratio = 0.3
    sampled_shape = (int(gaussians.xyz.shape[0]*sample_ratio), gaussians.xyz.shape[1])
    num_pts = sampled_shape[0]

    # Generate means
    min_values = np.min(gaussians.xyz, axis=0)
    max_values = np.max(gaussians.xyz, axis=0)
    xyz = np.random.uniform(min_values, max_values, size=sampled_shape).astype(np.float32)

    # Generate opacities
    opacities = np.ones((num_pts, 1)).astype(np.float32)

    # Generate scales
    scales = np.random.random((num_pts, 3)).astype(np.float32) * np.median(gaussians.scale)

    # Generate rotations
    rots = np.concatenate((np.ones((num_pts, 1)), np.zeros((num_pts, 3))), axis=1).astype(np.float32)

    # Generate colors
    rgb = np.random.random((num_pts, 3)).astype(np.float32)

    return GaussianData(xyz, rots, scales, opacities, rgb)

def load_ply(path):
    max_sh_degree = 3

    plydata = PlyData.read(path)

    # Extract xyz coordinates (means of the Gaussians)
    x = np.asarray(plydata.elements[0]["x"])
    y = np.asarray(plydata.elements[0]["y"])
    z = np.asarray(plydata.elements[0]["z"])
    xyz = np.stack((x, y, z), axis=1)
    xyz = xyz.astype(np.float32)

    # num_pts
    num_pts = xyz.shape[0]
    
    # Extract opacities
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    opacities = 1 / (1 + np.exp(-opacities))  # sigmoid
    opacities = opacities.astype(np.float32)

    # Extract scales
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((num_pts, len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    scales = np.exp(scales)
    scales = scales.astype(np.float32)

    # Extract rotations
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((num_pts, len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)

    # Extract directional coefficients
    features_dc = np.zeros((num_pts, 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    # Extract additional features related to spherical harmonics
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((num_pts, len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P, F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    # Concatenate directional coefficients and additional features to form final spherical harmonics 
    shs = np.concatenate([features_dc.reshape(-1, 3), features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)

    if "n_strands" in plydata['vertex'].data.dtype.names:
        head_avatar_constants = (plydata.elements[0]["n_strands"][0], plydata.elements[0]["n_gaussians_per_strand"][0])
    else:
        head_avatar_constants = (0, 0)
    
    return GaussianData(xyz, rots, scales, opacities, shs), head_avatar_constants

def load_input_ply(path):
    plydata = PlyData.read(path)

    # Extract xyz coordinates (means of the Gaussians)
    x = np.asarray(plydata.elements[0]["x"])
    y = np.asarray(plydata.elements[0]["y"])
    z = np.asarray(plydata.elements[0]["z"])
    xyz = np.stack((x, y, z), axis=1)
    xyz = xyz.astype(np.float32)

    # num_pts
    num_pts = xyz.shape[0]

    # Generate opacities
    opacities = np.ones((num_pts, 1)).astype(np.float32)

    # Generate scales
    scales = np.random.random((num_pts, 3)).astype(np.float32) * 0.5

    # Generate rotations
    rots = np.concatenate((np.ones((num_pts, 1)), np.zeros((num_pts, 3))), axis=1).astype(np.float32)

    # Extract colors
    r = np.asarray(plydata.elements[0]["red"]) / 255
    g = np.asarray(plydata.elements[0]["green"]) / 255
    b = np.asarray(plydata.elements[0]["blue"]) / 255
    rgb = np.stack((r, g, b), axis=1).astype(np.float32)

    return GaussianData(xyz, rots, scales, opacities, rgb)


if __name__ == "__main__":
    gs = load_ply("D:\\Daniel\\Masters\\Term 2\\Practical Machine Learning\\Models\\bicycle\\point_cloud\\iteration_7000\\point_cloud.ply")
    a = gs.flat()
    print(a.shape)
