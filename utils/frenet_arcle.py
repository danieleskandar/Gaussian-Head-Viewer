# frenet can be parametrized using arclength or 3d points.
# since the points are uniformly distributed both are ok

import numpy as np
import argparse
import os 
try:
    from utils import util_gau
except:
    import util_gau

def get_hair_points(xyz, rot, scale, n_strands, n_gaussians_per_strand, n_hair_gaussians):
    if n_strands == 0:
        return np.array([]), np.array([]),

    strands = np.zeros((n_strands, n_gaussians_per_strand+1, 3))
    strands_xyz = xyz[:n_hair_gaussians].reshape(n_strands, n_gaussians_per_strand, -1)
    strands_rot = rot[:n_hair_gaussians].reshape(n_strands, n_gaussians_per_strand, -1)
    strands_scale = scale[:n_hair_gaussians].reshape(n_strands, n_gaussians_per_strand, -1)

    w, x, y, z = strands_rot.transpose(2, 0, 1)
    global_x_displacement = np.array([1. - 2. * (y * y + z * z), 2. * (x * y + w * z), 2. * (x * z - w * y)]).transpose(1,2,0)

    displacements = 0.5*strands_scale*global_x_displacement
    strands[:,0] = strands_xyz[:,0] - displacements[:,0]
    strands[:,1:] = strands_xyz + displacements

    # Mean x-displacement of the hair gaussians after first couple gaussians
    start = n_gaussians_per_strand//10
    mean_last_disps = np.mean(global_x_displacement[:,start:], axis=1)

    # Orthogonal vectors which lie on the plane perpendicular to hair
    normals = np.zeros_like(mean_last_disps)
    normals[:, 0], normals[:, 1] = mean_last_disps[:, 1], -mean_last_disps[:, 0]
    binormals = np.cross(mean_last_disps, normals)
    normals /= np.linalg.norm(normals, axis=1)[:,np.newaxis]
    binormals /= np.linalg.norm(binormals, axis=1)[:,np.newaxis]
    disps = np.stack((normals, binormals))
    return strands, disps

def get_curls(amp, freq, hair_normals, n_gaussians_per_strand, n_strands):
    t = np.linspace(0, 2, n_gaussians_per_strand+1)[:,np.newaxis].T

    # Fixing random seed for future random initial frequency and overall noise
    np.random.seed(0)
    random_init_freq = np.random.uniform(low=0, high=2*np.pi, size=(n_strands,1))
    # Parameter t with random initial values so it doesn't look too uniform
    t_strands = t+random_init_freq
    # Multiplier to t value so it curls either way
    random_dir = np.random.choice([-1, 1], size=(n_strands,1))

    # Quadratic so hair roots are not displaced
    amplitude = (t**2*amp)[:,:,np.newaxis]
    # Sine and cosine to get coil shaped curls and not one-dimensional
    sin_wave = amplitude*np.sin(random_dir*(np.pi * freq * t + t_strands))[:,:,np.newaxis]
    cos_wave = amplitude*np.cos(random_dir*(np.pi * freq * t + t_strands))[:,:,np.newaxis]

    sin_noise = cos_noise = 0
    if amp!=0:
        sin_noise = np.random.normal(0, amp/30, size=sin_wave.shape)
        cos_noise = np.random.normal(0, amp/30, size=cos_wave.shape)

    # The curls are applied along the two vectors that form the plane perpendicular to the hair
    global_nudging = (sin_wave+sin_noise)*hair_normals[0][:,np.newaxis] + (cos_wave+cos_noise)*hair_normals[1][:,np.newaxis]
    return global_nudging

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
    return midpoints, x_scales

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
    amps_freqs = np.vstack((amps, freqs))
    rxyzs = np.zeros((ncurls, ncurls,n_strands*n_gaussians_per_strand, strands_rot.shape[2]+strands_xyz.shape[2]+1), dtype=np.float16)

    for i, amp in enumerate(amps):
        for j, freq in enumerate(freqs):
            global_nudging = get_curls(amp, freq, normals, n_gaussians_per_strand, n_strands)
            new_points = points+global_nudging
            xyz, xscale = calculate_pts_scal(new_points)
            rot = calculate_rot_quat(new_points)

            rot, xyz, scale = np.float16(rot), np.float16(xyz), np.float16(xscale)
            rxyzs[i,j,:,:4] = rot.reshape(-1,4)
            rxyzs[i,j,:,4:7] = xyz.reshape(-1,3)
            rxyzs[i,j,:,7] = scale.flatten()

    np.save(os.path.join(os.path.dirname(head_file), 'rxyzs.npy'), rxyzs)
    np.save(os.path.join(os.path.dirname(head_file), 'amps_freqs.npy'), amps_freqs)

        
def main(args):
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    
    if args.n_samples != 0:
        print(f'Calculating {str(args.n_samples)} curls with amplitude 0 to {args.max_amp} and frequency 0 to {args.max_freq}.')
        calculate_frenet_curls(args.input, args.n_samples, args.max_amp, args.max_freq)
        return 0

    if args.input.endswith('.npy'):
        print('Frenet frames are calculated and saved for *single* frame.')
        calculate_frenet_frame_t(args.input, args)
    else:
        print('Frenet frames are calculated and saved for all frames.')
        frames = os.listdir(args.input)
        for i in range(len(frames)):
            calculate_frenet_frame_t(args.input + frames[i], args)


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

