# frenet can be parametrized using arclength or 3d points.
# since the points are uniformly distributed both are ok


import numpy as np
import argparse
import os 

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

def calculate_scale(points):
    num_segments = len(points) - 1
    scales = []
    for i in range(num_segments):
        scale = np.linalg.norm(points[i] - points[i+1])
        scales.append(np.array([scale, scale/10.0, scale/10.0]))

    scales = np.array(scales)
    return scales

def calculate_scale_opt(points):
    scales = np.linalg.norm(points[:-1]-points[1:], axis=1)
    return np.block([[scales], [scales/10], [scales/10]]).T

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


def calculate_tnb_frames_opt(points):
    # Initialize T, N, B arrays
    T = np.zeros_like(points)
    N = np.zeros_like(points)

    # Approximate Tangent Vectors
    T[1:-1] = points[2:]-points[:-2]
    
    # Handle start/end points (simple extrapolation)
    T[0] = points[1] - points[0]
    T[-1] = points[-1] - points[-2]

    T_norms = np.linalg.norm(T, axis=1)
    T_norms[T_norms<1e-8] = 1
    T /= T_norms[:,np.newaxis]

    # Approximate Normal and Binormal Vectors with Fallback
    N[:-1] = T[1:]-T[:-1]
    N[-1] = N[-2]

    N_norms = np.linalg.norm(N, axis=1)
    N_norms[N_norms<1e-8] = 1
    N /= N_norms[:,np.newaxis]
    N[1:][N_norms[1:]<1e-8] = N[:-1][N_norms[:-1]<1e-8]

    B = np.cross(T, N)
    B_norms = np.linalg.norm(B, axis=1)
    B_norms[B_norms<1e-8] = 1
    B /= B_norms[:,np.newaxis]
    B[1:][B_norms[1:]<1e-8] = B[:-1][B_norms[:-1]<1e-8]

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

def interpolate_tnb_linear_opt(T, N, B, points): # shape is 32, 3
    midpoints = (points[:-1] + points[1:]) / 2
    interpolated_T = (T[:-1] + T[1:]) / 2
    interpolated_N = (N[:-1] + N[1:]) / 2
    interpolated_B = (B[:-1] + B[1:]) / 2

    # Normalize the interpolated vectors
    interpolated_T /= np.linalg.norm(interpolated_T, axis=1)[:,np.newaxis] + 1e-8
    interpolated_N /= np.linalg.norm(interpolated_N, axis=1)[:,np.newaxis] + 1e-8
    interpolated_B /= np.linalg.norm(interpolated_B, axis=1)[:,np.newaxis] + 1e-8

    return interpolated_T, interpolated_N, interpolated_B, midpoints

# strands has shape (#strands, 32, 3). iterates over all strands
def calculate_frenet_frame_t_npy(hair_strands):
    groom_scales = []
    groom_R = []
    groom_midpoints = []
    for i, s in enumerate(hair_strands):
        T, N, B = calculate_tnb_frames_opt(s)
        scales = calculate_scale_opt(s)
        
        # Interpolate TNB and calculate midpoints
        interpolated_T, interpolated_N, interpolated_B, midpoints = interpolate_tnb_linear_opt(T, N, B, s)
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

def main(args):
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

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