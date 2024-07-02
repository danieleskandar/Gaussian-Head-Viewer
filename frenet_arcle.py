# frenet can be parametrized using arclength or 3d points.
# since the points are uniformly distributed both are ok

import multiprocessing as mp
import numpy as np
import argparse
import os 
import util_gau

N_HAIR_STRANDS_dict = {"colored":0, "large":4000, "medium":150, "small":10, "dense": 12000}
N_HAIR_STRANDS = 0
N_GAUSSIANS_PER_STRAND = 0
N_HAIR_GAUSSIANS = 0

# use this to get rid of the nan values, if any
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def rotmats2qvecs(R):
    Rxx, Ryx, Rzx = R[:, :, 0, 0], R[:, :, 1, 0], R[:, :, 2, 0]
    Rxy, Ryy, Rzy = R[:, :, 0, 1], R[:, :, 1, 1], R[:, :, 2, 1]
    Rxz, Ryz, Rzz = R[:, :, 0, 2], R[:, :, 1, 2], R[:, :, 2, 2]
    
    # Construct the K matrices for all matrices
    K = np.empty((R.shape[0], R.shape[1], 4, 4))
    K[:, :, 0, 0] = Rxx - Ryy - Rzz
    K[:, :, 0, 1] = 0
    K[:, :, 0, 2] = 0
    K[:, :, 0, 3] = 0
    
    K[:, :, 1, 0] = Ryx + Rxy
    K[:, :, 1, 1] = Ryy - Rxx - Rzz
    K[:, :, 1, 2] = 0
    K[:, :, 1, 3] = 0
    
    K[:, :, 2, 0] = Rzx + Rxz
    K[:, :, 2, 1] = Rzy + Ryz
    K[:, :, 2, 2] = Rzz - Rxx - Ryy
    K[:, :, 2, 3] = 0
    
    K[:, :, 3, 0] = Ryz - Rzy
    K[:, :, 3, 1] = Rzx - Rxz
    K[:, :, 3, 2] = Rxy - Ryx
    K[:, :, 3, 3] = Rxx + Ryy + Rzz
    
    K /= 3.0
    
    _, eigvecs = np.linalg.eigh(K)    
    qvecs = eigvecs[:, :, [3, 0, 1, 2], -1]
    qvecs[qvecs[:, :, 0] < 0] *= -1
    return qvecs

def calculate_scale(points):
    num_segments = len(points) - 1
    scales = []
    for i in range(num_segments):
        scale = np.linalg.norm(points[i] - points[i+1])
        scales.append(np.array([scale, scale/10.0, scale/10.0]))

    scales = np.array(scales)
    return scales

def calculate_tnb_frames(points):
    """Calculates Frenet-Serret (TNB) frames for a discrete curve without smoothing.

    Args:
        points: List of 3D point coordinates.

    Returns:
        T, N, B: Lists of T, N, B vectors for each point.
    """

    points = np.array(points)
    num_points = len(points)

    # Initialize T, N, B arrays
    T = np.zeros_like(points)
    N = np.zeros_like(points)
    B = np.zeros_like(points)

    # Approximate Tangent Vectors
    for i in range(1, num_points - 1):
        T[i] = points[i + 1] - points[i - 1]

    # Handle start/end points (simple extrapolation)
    T[0] = points[1] - points[0]
    T[-1] = points[-1] - points[-2]

    for i in range(num_points):
        if np.linalg.norm(T[i]) < 1e-8:  # Check for near-zero norm
            print("Near-zero norm at index:", i)
            assert False
            #T[i] = T[i - 1]  # Fallback to previous tangent
        else:
            T[i] /= np.linalg.norm(T[i])

    # Approximate Normal and Binormal Vectors with Fallback
    for i in range(num_points):
        if i < num_points - 1:
            N[i] = T[i + 1] - T[i]
        else:
            N[i] = N[i - 1]
 
        if np.linalg.norm(N[i]) < 1e-8:  # Check for near-zero norm
            print("Near-zero norm at index for normals:", i)
            #assert False
            N[i] = N[i - 1]  # Fallback to previous normal
        else:
            N[i] /= np.linalg.norm(N[i])


        B[i] = np.cross(T[i], N[i])
        if np.linalg.norm(B[i]) < 1e-8:
            B[i] =  B[i-1]
        else:
            B[i] /= np.linalg.norm(B[i])

    return T, N, B
    
def interpolate_tnb_linear(T, N, B, points):
    num_segments = len(points) - 1
    interpolated_T, interpolated_N, interpolated_B = [], [], []
    midpoints = []

    for i in range(num_segments):
        midpoint = (points[i] + points[i + 1]) / 2
        interpolated_T.append((T[i] + T[i + 1]) / 2)
        interpolated_N.append((N[i] + N[i + 1]) / 2)
        interpolated_B.append((B[i] + B[i + 1]) / 2)
        midpoints.append(midpoint)

    # Normalize the interpolated vectors
    interpolated_T = [v / (np.linalg.norm(v) + 1e-8) for v in interpolated_T]
    interpolated_N = [v / (np.linalg.norm(v) + 1e-8) for v in interpolated_N]
    interpolated_B = [v / (np.linalg.norm(v) + 1e-8) for v in interpolated_B]

    midpoints = np.array(midpoints)
    return interpolated_T, interpolated_N, interpolated_B, midpoints

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

def calculate_rot_quat(hair_strands):
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

    # Quaternion representing the rotation of the 1st canonical vector (1,0,0) to the tangents. 
    # Exploits simple vector and the already calculated normals
    quats_T = np.dstack((1 + interpolated_T[:,:,0], interpolated_N))
    quats_T /= np.linalg.norm(quats_T, axis=2)[:,:,np.newaxis]

    w, x, y, z = quats_T.transpose(2, 0, 1)
    # 2nd canonical vector is rotated by quats_T. Used SymPy to exploit simplicity of canonical vectors.
    normals = np.array([-2*w*z + 2*x*y, w**2 - x**2 + y**2 - z**2, 2*w*x + 2*y*z]).transpose(1,2,0)
    quats_N = np.dstack((1 + np.einsum('ijk,ijk->ij', normals, interpolated_N), np.cross(normals, interpolated_N)))
    quats_N /= np.linalg.norm(quats_N, axis=2)[:,:,np.newaxis]

    # Getting the rotation quaternion from rotating with quats_T and then quats_N
    quats_NT = quaternions_multiply(quats_N, quats_T)
    w, x, y, z = quats_NT.transpose(2, 0, 1)
    # 3rd canonical vector is rotated by quats_T. Used SymPy to exploit simplicity of canonical vectors.
    binormals = np.array([2*w*y + 2*x*z, -2*w*x + 2*y*z, w**2 - x**2 - y**2 + z**2]).transpose(1,2,0)
    quats_B = np.dstack((1 + np.einsum('ijk,ijk->ij', binormals, interpolated_B), np.cross(binormals, interpolated_B)))
    quats_B /= np.linalg.norm(quats_B, axis=2)[:,:,np.newaxis]

    # Getting the final rotation quaternion
    quats_BNT = quaternions_multiply(quats_B, quats_NT)
    quats_BNT[quats_BNT[:,:,0]<0] *= -1

    return quats_BNT

# strands has shape (#strands, 32, 3). iterates over all strands
def calculate_frenet_frame_t_opt(hair_strands):

    T = np.zeros_like(hair_strands)
    # Approximate Tangent Vectors
    T[:,1:-1,:] = hair_strands[:,2:,:]-hair_strands[:,:-2,:]

    # Handle start/end points (simple extrapolation)
    T[:,0,:] = hair_strands[:,1,:] - hair_strands[:,0,:]
    T[:,-1,:] = hair_strands[:,-1,:] - hair_strands[:,-2,:]
    # Normalize the tangent vectors and if too small fallback
    T = normalize_or_fallback(T)

    # Approximate Normal and Binormal Vectors with Fallback
    N = np.zeros_like(hair_strands)
    N[:,:,0] = T[:,:,1]
    N[:,:,1] = -T[:,:,0]
    B = np.cross(T, N)

    # Interpolate and normalize
    interpolated_T = interpolate_and_normalize(T)
    interpolated_N = interpolate_and_normalize(N)
    interpolated_B = interpolate_and_normalize(B)

    R = np.stack((interpolated_T, interpolated_N, interpolated_B), axis=-1).transpose((0, 1, 3, 2))
    qvecs = rotmats2qvecs(R)
    
    midpoints = (hair_strands[:,:-1,:] + hair_strands[:,1:,:]) / 2

    x_scales = np.linalg.norm(hair_strands[:,:-1,:]-hair_strands[:,1:,:], axis=2)
    scales = np.dstack([x_scales, x_scales/10, x_scales/10])

    return midpoints, qvecs, scales

def parallel_calculate_quats(input_array):
    num_cpus = mp.cpu_count()
    N = input_array.shape[0]
    chunk_size = (N + num_cpus - 1) // num_cpus  # Compute chunk size
    chunks = [input_array[i:min(i + chunk_size, N)] for i in range(0, N, chunk_size)]
    
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(calculate_rot_quat, chunks)
    
    quats = np.concatenate(results, axis=0)
    return quats

def calculate_frenet_frame_t_non(hair_strands):
    groom_scales = []
    groom_R = []
    groom_midpoints = []
    for i, s in enumerate(hair_strands):
        T, N, B = calculate_tnb_frames(s)
        scales = calculate_scale(s)
        
        # Interpolate TNB and calculate midpoints
        interpolated_T, interpolated_N, interpolated_B, midpoints = interpolate_tnb_linear(T, N, B, s)
        R_matrices = []
        for i in range(len(midpoints)):
            R = np.column_stack((interpolated_T[i], interpolated_N[i], interpolated_B[i]))
            R_matrices.append(rotmat2qvec(R))

        R_matrices = np.array(R_matrices)

        groom_R.append(R_matrices)
        groom_scales.append(scales)
        groom_midpoints.append(midpoints)

    groom_R = np.array(groom_R)
    groom_scales = np.array(groom_scales)
    groom_midpoints = np.array(groom_midpoints)

    return groom_midpoints, groom_R, groom_scales

def calculate_frenet_frame_t(inp_strands, args):
    hair_strand_points = np.load(inp_strands)
    print('Strands shape:', hair_strand_points.shape)

    groom_scales = []
    groom_R = []
    groom_midpoints = []
    for i, s in enumerate(hair_strand_points):
        T, N, B = calculate_tnb_frames(s)
        scales = calculate_scale(s)
        
        
        # Interpolate TNB and calculate midpoints
        interpolated_T, interpolated_N, interpolated_B, midpoints = interpolate_tnb_linear(T, N, B, s)
        R_matrices = []
        for i in range(len(midpoints)):
            R = np.column_stack((interpolated_T[i], interpolated_N[i], interpolated_B[i]))
            if args.rot_format == 'quat':
                R_matrices.append(rotmat2qvec(R))
            else:
                R_matrices.append(R)

        R_matrices = np.array(R_matrices)

        groom_R.append(R_matrices)
        groom_scales.append(scales)
        groom_midpoints.append(midpoints)
    

    groom_R = np.array(groom_R)
    groom_scales = np.array(groom_scales)
    groom_midpoints = np.array(groom_midpoints)

    print('Rotation Matrices:', groom_R.shape)
    print('Number of scales calculated:', groom_scales.shape)
    print('Number of midpoints:', groom_midpoints.shape)

    mean_frenet = inp_strands.replace('.npy', '_mean_frenet.npy')
    rot_frenet = inp_strands.replace('.npy', '_rot_frenet.npy')
    scale_frenet = inp_strands.replace('.npy', '_scale_frenet.npy')

    np.save(mean_frenet, groom_midpoints)
    np.save(rot_frenet, groom_R)
    np.save(scale_frenet, groom_scales)

def calculate_frenet_curls(head_file, ncurls, max_amp, max_freq):
    global N_HAIR_STRANDS, N_GAUSSIANS_PER_STRAND, N_HAIR_GAUSSIANS

    file_path = f"./models/head ({head_file})/point_cloud/iteration_30000/point_cloud.ply"
    head_avatar = util_gau.load_ply(file_path)
    N_HAIR_STRANDS = N_HAIR_STRANDS_dict[head_file]
    N_GAUSSIANS_PER_STRAND = 31
    N_HAIR_GAUSSIANS = N_HAIR_STRANDS * N_GAUSSIANS_PER_STRAND

    strands_xyz = head_avatar.xyz[:N_HAIR_GAUSSIANS].reshape(N_HAIR_STRANDS, N_GAUSSIANS_PER_STRAND, -1)
    strands_rot = head_avatar.rot[:N_HAIR_GAUSSIANS].reshape(N_HAIR_STRANDS, N_GAUSSIANS_PER_STRAND, -1)
    strands_scale = head_avatar.scale[:N_HAIR_GAUSSIANS].reshape(N_HAIR_STRANDS, N_GAUSSIANS_PER_STRAND, -1)

    points, normals = get_hair_points(strands_xyz, strands_rot, strands_scale)
    amps = np.linspace(max_amp, 0, ncurls, endpoint=False)[::-1]
    freqs= np.linspace(max_freq, 0, ncurls, endpoint=False)[::-1]
    for amp in amps:
        for freq in freqs:
            global_nudging = get_curls(amp, freq, normals)
            new_points = points+global_nudging
            rots = calculate_rot_quat(new_points)
            
            newpath = f"./models/head ({head_file})/rots/{amp:.8f}/"
            if not os.path.exists(newpath):
                os.makedirs(newpath)

            np.save(f"./models/head ({head_file})/rots/{amp:.8f}/{freq:.8f}.npy",  rots)
        
def get_hair_points(xyz, rot, scale):
    global N_HAIR_STRANDS, N_GAUSSIANS_PER_STRAND, N_HAIR_GAUSSIANS
    strands = np.zeros((N_HAIR_STRANDS, N_GAUSSIANS_PER_STRAND+1, 3))
    strands_xyz = xyz[:N_HAIR_GAUSSIANS].reshape(N_HAIR_STRANDS, N_GAUSSIANS_PER_STRAND, -1)
    strands_rot = rot[:N_HAIR_GAUSSIANS].reshape(N_HAIR_STRANDS, N_GAUSSIANS_PER_STRAND, -1)
    strands_scale = scale[:N_HAIR_GAUSSIANS].reshape(N_HAIR_STRANDS, N_GAUSSIANS_PER_STRAND, -1)

    w, x, y, z = strands_rot.transpose(2, 0, 1)
    global_x_displacement = np.array([1. - 2. * (y * y + z * z), 2. * (x * y + w * z), 2. * (x * z - w * y)]).transpose(1,2,0)
    
    displacements = 0.5*strands_scale*global_x_displacement 
    strands[:,0] = strands_xyz[:,0] - displacements[:,0]
    strands[:,1:] = strands_xyz + displacements

    # Mean x-displacement of the hair gaussians after first couple gaussians
    start = N_GAUSSIANS_PER_STRAND//10
    mean_last_disps = np.mean(global_x_displacement[:,start:], axis=1)

    # Orthogonal vectors which lie on the plane perpendicular to hair
    normals = np.zeros_like(mean_last_disps)
    normals[:, 0], normals[:, 1] = mean_last_disps[:, 1], -mean_last_disps[:, 0]
    binormals = np.cross(mean_last_disps, normals)
    normals /= np.linalg.norm(normals, axis=1)[:,np.newaxis]
    binormals /= np.linalg.norm(binormals, axis=1)[:,np.newaxis]
    disps = np.stack((normals, binormals))
    return strands, disps

def get_curls(amp, freq, hair_normals):
    global N_HAIR_STRANDS, N_GAUSSIANS_PER_STRAND, N_HAIR_GAUSSIANS
    t = np.linspace(0, 2, N_GAUSSIANS_PER_STRAND+1)[:,np.newaxis].T
    
    # Fixing random seed for future random initial frequency and overall noise
    np.random.seed(0)
    random_init_freq = np.random.uniform(low=0, high=2*np.pi, size=(N_HAIR_STRANDS,1))
    # Parameter t with random initial values so it doesn't look too uniform
    t_strands = t+random_init_freq
    # Multiplier to t value so it curls either way
    random_dir = np.random.choice([-1, 1], size=(N_HAIR_STRANDS,1))

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

def main(args):
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    
    if args.ncurls != 0:
        print(f'Calculating {str(args.ncurls)} curls with amplitude 0 to {args.max_amp} and frequency 0 to {args.max_freq}.')
        calculate_frenet_curls(args.inp_strands, args.ncurls, args.max_amp, args.max_freq)
        return 0

    if args.inp_strands.endswith('.npy'):
        calculate_frenet_frame_t(args.inp_strands, args)
        print('Frenet frames are calculated and saved for *single* frame.')
    else:
        frames = os.listdir(args.inp_strands)
        for i in range(len(frames)):
            calculate_frenet_frame_t(args.inp_strands + frames[i], args)
        print('Frenet frames are calculated and saved for all frames.')


    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    #parser.add_argument('--inp_strands', default='/home/bkabadayi/data2tb/tmp_experiments/phyfaceavatar/sparse_hair_Actor03/guided_2_k10_p32_dist.npy', type=str)
    parser.add_argument('--inp_strands', default='/home/bkabadayi/data2tb/rot_set/data/disp/320_to_320/', type=str)
    parser.add_argument('--rot_format', choices=['quat', 'mat'], help='if frenet is for init, use quat, for animation use matrix', default='mat', type=str)
    parser.add_argument('--ncurls', default=0, type=int)
    parser.add_argument('--max_amp', default=0.025, type=float)
    parser.add_argument('--max_freq', default=3, type=float)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)



# # Generate 10 points along a helix for a more meaningful example
# t = np.linspace(0, 4 * np.pi, 10)  # Increase the number of points for smoother curve
# x = np.sin(t)
# y = np.cos(t)
# z = t
# hair_strand_points = np.vstack((x, y, z)).T

# # 
# hear_mean_p = '/home/bkabadayi/data2tb/rot_set/data/meshes/frame_49.npy'
# hair_strand_points = np.load(hear_mean_p).reshape(10, 32, 3)[0]
# print('Strands shape:', hair_strand_points.shape)


# # Calculate TNB frames
# T, N, B = calculate_tnb_frames(hair_strand_points)
# all_scales = calculate_scale(hair_strand_points)


# # Interpolate TNB and calculate midpoints
# interpolated_T, interpolated_N, interpolated_B, midpoints = interpolate_tnb_linear(T, N, B, hair_strand_points)

# print("Interpolated Tangent Vectors (T):")
# print(interpolated_T)
# print("Interpolated Normal Vectors (N):")
# print(interpolated_N)
# print("Interpolated Binormal Vectors (B):")
# print(interpolated_B)
# print("Midpoints:")
# print(midpoints)

# R_matrices = []
# for i in range(len(midpoints)):
#     R = np.column_stack((interpolated_T[i], interpolated_N[i], interpolated_B[i]))
#     R_matrices.append(R)

# R_matrices = np.array(R_matrices)

# print('Rotation Matrices:', R_matrices.shape)
# print('Number of scales calculated:', all_scales.shape)
# print('Number of midpoints:', midpoints.shape)


# import matplotlib.pyplot as plt

# # Create 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the curve
# for i in range(len(hair_strand_points) - 1):
#     ax.plot(
#         [hair_strand_points[i][0], hair_strand_points[i + 1][0]],
#         [hair_strand_points[i][1], hair_strand_points[i + 1][1]],
#         [hair_strand_points[i][2], hair_strand_points[i + 1][2]],
#         'b-'  # Blue line
#     )

# # Plot the original points with labels
# for i, point in enumerate(hair_strand_points):
#     ax.scatter(*point, c='r', marker='o')  # Red dots
#     ax.text(*point, s=str(i), fontsize=10)  # Add index labels

# # Plot the TNB frames at original points (solid lines)
# scale_factor = 0.01  # Adjust as needed
# for i, point in enumerate(hair_strand_points):
#     ax.quiver(
#         point[0], point[1], point[2],  # Starting point
#         T[i][0], T[i][1], T[i][2],  # T direction
#         color='r', length=scale_factor, normalize=True,
#     )
#     ax.quiver(
#         point[0], point[1], point[2],  # Starting point
#         N[i][0], N[i][1], N[i][2],  # N direction
#         color='g', length=scale_factor, normalize=True,
#     )
#     ax.quiver(
#         point[0], point[1], point[2],  # Starting point
#         B[i][0], B[i][1], B[i][2],  # B direction
#         color='b', length=scale_factor, normalize=True,
#     )

# # Plot the interpolated frames (TNB) at midpoints (dashed lines)
# for i, midpoint in enumerate(midpoints):
    
#     # ax.quiver(
#     #     midpoint[0], midpoint[1], midpoint[2],  # Starting point (midpoint)
#     #     interpolated_T[i][0], interpolated_T[i][1], interpolated_T[i][2],  # T direction
#     #     color='r', linestyle='--', length=scale_factor, normalize=True,
#     # )
#     ax.quiver(
#         midpoint[0], midpoint[1], midpoint[2],  # Starting point (midpoint)
#         interpolated_N[i][0], interpolated_N[i][1], interpolated_N[i][2],  # N direction
#         color='g', linestyle='--', length=scale_factor, normalize=True,
#     )
#     # ax.quiver(
#     #     midpoint[0], midpoint[1], midpoint[2],  # Starting point (midpoint)
#     #     interpolated_B[i][0], interpolated_B[i][1], interpolated_B[i][2],  # B direction
#     #     color='b', linestyle='--', length=scale_factor, normalize=True,
#     # )

# # Label axes and add title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Hair Strand Curve, Points, and TNB Frames')

# # Show the plot
# plt.show()