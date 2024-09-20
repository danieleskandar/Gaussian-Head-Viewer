# frenet can be parametrized using arclength or 3d points.
# since the points are uniformly distributed both are ok

import numpy as np
import argparse
import os 
import utils.util_gau
from main import *

def TNB2qvecs(T, N, B):
    # Quaternion representing the rotation of the 1st canonical vector (1,0,0) to the tangents. 
    # Exploits simple vector and the already calculated normals
    quats_T = np.dstack((1 + T[:,:,0], N))
    quats_T /= np.linalg.norm(quats_T, axis=2)[:,:,np.newaxis]

    w, x, y, z = quats_T.transpose(2, 0, 1)
    # 2nd canonical vector is rotated by quats_T. Used SymPy to exploit simplicity of canonical vectors.
    normals = np.array([-2*w*z + 2*x*y, w**2 - x**2 + y**2 - z**2, 2*w*x + 2*y*z]).transpose(1,2,0)
    quats_N = np.dstack((1 + np.einsum('ijk,ijk->ij', normals, N), np.cross(normals, N)))
    quats_N /= np.linalg.norm(quats_N, axis=2)[:,:,np.newaxis]

    # Getting the rotation quaternion from rotating with quats_T and then quats_N
    quats_NT = quaternions_multiply(quats_N, quats_T)
    w, x, y, z = quats_NT.transpose(2, 0, 1)
    # 3rd canonical vector is rotated by quats_T. Used SymPy to exploit simplicity of canonical vectors.
    binormals = np.array([2*w*y + 2*x*z, -2*w*x + 2*y*z, w**2 - x**2 - y**2 + z**2]).transpose(1,2,0)
    quats_B = np.dstack((1 + np.einsum('ijk,ijk->ij', binormals, B), np.cross(binormals, B)))
    quats_B /= np.linalg.norm(quats_B, axis=2)[:,:,np.newaxis]

    # Getting the final rotation quaternion
    quats_BNT = quaternions_multiply(quats_B, quats_NT)
    quats_BNT[quats_BNT[:,:,0]<0] *= -1

    return quats_BNT


# Normalizes the tangent vectors and if too small fallback
def normalize_or_fallback(vector):
    norms = np.linalg.norm(vector, axis=2)
    norms[norms<1e-8] = 1
    vector /= norms[:,:,np.newaxis]
    # Fallback to previous tangent
    vector[:,1:,:][norms[:,1:]<1e-8] = vector[:,:-1,:][norms[:,:-1]<1e-8]
    return vector

def interpolate_and_normalize(vector):
    interpolated_vector = (vector[:,:-1,:] + vector[:,1:,:]) / 2
    return normalize_or_fallback(interpolated_vector)

def quaternions_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0.transpose(2, 0, 1)
    w1, x1, y1, z1 = quaternion1.transpose(2, 0, 1)
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0]).transpose(1,2,0)

def calculate_pts_scal(hair_strands):
    midpoints = (hair_strands[:,:-1,:] + hair_strands[:,1:,:]) / 2
    x_scales = np.linalg.norm(hair_strands[:,:-1,:]-hair_strands[:,1:,:], axis=2)
    scales = np.dstack([x_scales, x_scales/10, x_scales/10])
    return midpoints, scales

def calculate_TNB(hair_strands):
    T = np.zeros_like(hair_strands)
    # Approximate Tangent Vectors
    T[:,1:-1,:] = hair_strands[:,2:,:]-hair_strands[:,:-2,:]

    # Handle start/end points (simple extrapolation)
    T[:,0,:] = hair_strands[:,1,:] - hair_strands[:,0,:]
    T[:,-1,:] = hair_strands[:,-1,:] - hair_strands[:,-2,:]
    # Normalize the tangent vectors and if too small fallback
    T = normalize_or_fallback(T)

    # Approximate Normal and Binormal Vectors
    N = np.zeros_like(hair_strands)
    N[:,:,1] = -T[:,:,2]
    N[:,:,2] = T[:,:,1]
    B = np.cross(T, N)

    # Interpolate and normalize
    interpolated_T = interpolate_and_normalize(T)
    interpolated_N = interpolate_and_normalize(N)
    interpolated_B = interpolate_and_normalize(B)    
    return interpolated_T, interpolated_N, interpolated_B

def calculate_rot_quat(hair_strands):
    T, N, B = calculate_TNB(hair_strands)
    return TNB2qvecs(T, N, B)


def calculate_frenet_frame_t(inp_strands, args):
    hair_strand_points = np.load(inp_strands)
    print('Strands shape:', hair_strand_points.shape)
    
    midpoints, scales = calculate_pts_scal(hair_strand_points)
    T, N, B = calculate_TNB(hair_strand_points)
    
    R = None
    if args.rot_format == 'quat':
        R = TNB2qvecs(T, N, B)
    else:
        R = np.stack((T, N, B))

    print('Rotation Matrices:', R.shape)
    print('Number of scales calculated:', scales.shape)
    print('Number of midpoints:', midpoints.shape)

    mean_frenet = inp_strands.replace('.npy', '_mean_frenet.npy')
    rot_frenet = inp_strands.replace('.npy', '_rot_frenet.npy')
    scale_frenet = inp_strands.replace('.npy', '_scale_frenet.npy')

    np.save(mean_frenet, midpoints)
    np.save(rot_frenet, R)
    np.save(scale_frenet, scales)

def calculate_frenet_curls(head_file, ncurls, max_amp, max_freq):
    head_avatar, head_avatar_constants = util_gau.load_ply(head_file)
    n_strands, n_gaussians_per_strand = head_avatar_constants
    n_hair_gaussians = n_strands*n_gaussians_per_strand

    if n_hair_gaussians == 0:
        return

    strands_xyz = head_avatar.xyz[:n_hair_gaussians].reshape(n_strands, n_gaussians_per_strand, -1)
    strands_rot = head_avatar.rot[:n_hair_gaussians].reshape(n_strands, n_gaussians_per_strand, -1)
    strands_scale = head_avatar.scale[:n_hair_gaussians].reshape(n_strands, n_gaussians_per_strand, -1)

    points, normals = get_hair_points(strands_xyz, strands_rot, strands_scale, n_strands, n_gaussians_per_strand, n_hair_gaussians)
    amps = np.linspace(max_amp, 0, ncurls, endpoint=False)[::-1]
    freqs= np.linspace(max_freq, 0, ncurls, endpoint=False)[::-1]
    for amp in amps:
        for freq in freqs:
            global_nudging = get_curls(amp, freq, normals, n_gaussians_per_strand, n_strands)
            new_points = points+global_nudging
            rots = calculate_rot_quat(new_points)
            
            amp_path = os.path.join(os.path.dirname(head_file), f'rots/{amp:.8f}')
            if not os.path.exists(amp_path):
                os.makedirs(amp_path)

            freq_path = os.path.join(amp_path, f'{freq:.8f}.npy')
            np.save(freq_path,  rots)
        
def main(args):
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    
    if args.n_samples != 0:
        print(f'Calculating {str(args.n_samples)} curls with amplitude 0 to {args.max_amp} and frequency 0 to {args.max_freq}.')
        calculate_frenet_curls(args.input, args.n_samples, args.max_amp, args.max_freq)
        return 0

    if args.input.endswith('.npy'):
        calculate_frenet_frame_t(args.input, args)
        print('Frenet frames are calculated and saved for *single* frame.')
    else:
        frames = os.listdir(args.input)
        for i in range(len(frames)):
            calculate_frenet_frame_t(args.input + frames[i], args)
        print('Frenet frames are calculated and saved for all frames.')


    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('input', type=str)
    parser.add_argument('--rot_format', choices=['quat', 'mat'], help='if frenet is for init, use quat, for animation use matrix', default='mat', type=str)
    parser.add_argument('--n_samples', default=0, type=int)
    parser.add_argument('--max_amp', default=0.025, type=float)
    parser.add_argument('--max_freq', default=3, type=float)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)

