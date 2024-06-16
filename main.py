import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio
import util_gau
import tkinter as tk
from tkinter import filedialog
import os
import sys
import argparse
import time
import frenet_arcle
from renderer_ogl import OpenGLRenderer, GaussianRenderBase, OpenGLRendererAxes

# Add the directory containing main.py to the Python path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


g_camera = util.Camera(720, 1280)
BACKEND_OGL=0
BACKEND_CUDA=1
BACKEND_OGL_AXES=2
g_renderer_list = [
    None, None, None # ogl, cuda, ogl_axes
]
g_renderer_idx = BACKEND_OGL
g_renderer: GaussianRenderBase = g_renderer_list[g_renderer_idx]
g_scale_modifier = 1.
g_auto_sort = True
g_show_control_win = True
g_show_help_win = True
g_show_camera_win = True
g_render_mode_tables = ["Ray", "Gaussian Ball", "Flat Ball", "Billboard", "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 8

###########
# Constants
###########

CLICK_THRESHOLD = 0.2
DISPLACEMENT_FACTOR = 1.5

N_GAUSSIANS = 0
N_HAIR_STRANDS = 10
N_GAUSSIANS_PER_STRAND = 31
N_HAIR_GAUSSIANS = N_HAIR_STRANDS * N_GAUSSIANS_PER_STRAND
g_max_cutting_distance = 0.2
head_file = "small"

############################
# Mouse Controller Variables
############################

right_click_start_time = None
left_click_start_time = None

###################
# Utility Functions
###################
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

########################
# Head Avatars Variables
########################
gaussians = util_gau.naive_gaussian()
g_show_head_avatars_win = True
g_head_avatar_checkboxes = []
g_head_avatars = []
g_head_avatar_means = []
g_hair_points = []
g_z_plane = 1
g_z_max = 1
g_z_min = -1
g_empty_gaussian = util_gau.GaussianData(np.empty((1, 3)), np.empty((1, 4)), np.empty((1, 3)), np.empty((1, 3)), np.empty((1, 3)))
g_cutting_mode = False
g_coloring_mode = False
g_invert_z_plane = False

######################
# Head Avatars Actions
######################
def open_head_avatar_ply():
    global gaussians, N_GAUSSIANS, g_z_min, g_z_max, g_z_plane

    file_path = filedialog.askopenfilename(
        title="open ply",
        initialdir = f"./models/head ({head_file})/point_cloud/iteration_30000",
        filetypes=[('ply file', '.ply')]
    )
    if file_path:
        try:
            # Load head avatar
            head_avatar = util_gau.load_ply(file_path)

            # Fill controller arrays
            g_head_avatars.append(head_avatar)
            g_head_avatar_means.append(np.mean(head_avatar.xyz, axis=0))
            g_head_avatar_checkboxes.append(True)
            g_hair_points.append(get_hair_points(head_avatar.xyz, head_avatar.rot, head_avatar.scale))
            g_show_hair.append(True)
            g_show_head.append(True)
            g_hair_color.append([1, 0, 0])
            g_head_color.append([1, 1, 1])
            g_show_hair_color.append(False)
            g_show_head_color.append(False)
            g_hair_scale.append(1)
            g_wave_frequency.append(0)
            g_wave_amplitude.append(0)
            g_frame.append(0)
            
            if len(g_head_avatars) == 1:
                # Initialize global variables
                g_z_min = np.min(head_avatar.xyz[:, 2])
                g_z_max = np.max(head_avatar.xyz[:, 2])                
                g_z_plane = g_z_max
                N_GAUSSIANS = head_avatar.xyz.shape[0]
                g_renderer.update_N_GAUSSIANS(N_GAUSSIANS)

                # Append head avatar to the gaussians object sent to the shader
                xyz, rot, scale, opacity, sh = head_avatar.get_data()
                gaussians.xyz = xyz
                gaussians.rot = rot
                gaussians.scale = scale
                gaussians.opacity = opacity
                gaussians.sh = sh
            else:
                # Append head avatar to the gaussians object sent to the shader
                gaussians.xyz = np.vstack([gaussians.xyz, head_avatar.xyz]).astype(np.float32)
                gaussians.rot = np.vstack([gaussians.rot, head_avatar.rot]).astype(np.float32)
                gaussians.scale = np.vstack([gaussians.scale, head_avatar.scale]).astype(np.float32)
                gaussians.opacity = np.vstack([gaussians.opacity, head_avatar.opacity]).astype(np.float32)
                gaussians.sh = np.vstack([gaussians.sh, head_avatar.sh]).astype(np.float32)

        except RuntimeError as e:
            pass

##################################
# Head Avatar Controller Variables
##################################
g_show_head_avatar_controller_win = True
g_selected_head_avatar_index = -1
g_selected_head_avatar_name = "None"
g_show_hair = []
g_show_head = []
g_hair_color = []
g_head_color = []
g_show_hair_color = []
g_show_head_color = []
g_hair_scale = []
g_wave_frequency = []
g_wave_amplitude = []
g_frame = []

################################
# Head Avatar Controller Actions
################################
def select_closest_head_avatar():
    global g_selected_head_avatar_index, g_selected_head_avatar_name

    if len(g_head_avatars) == 0 or np.sum(g_head_avatar_checkboxes) == 0:
        g_selected_head_avatar_index = -1
        g_selected_head_avatar_name = "None"
        return

    # Get means of displayed head avatars
    avatar_means = np.vstack([g_head_avatar_means[i] for i in range(len(g_head_avatars)) if g_head_avatar_checkboxes[i]])

    # Get mouse 3D position
    mouse_pos_2d = imgui.get_io().mouse_pos
    mouse_pos_3d = util.glhUnProjectf(mouse_pos_2d.x, mouse_pos_2d.y, 1, g_camera.get_view_matrix(), g_camera.get_project_matrix(), gl.glGetIntegerv(gl.GL_VIEWPORT))

    # Compute ray direction
    ray_direction = mouse_pos_3d - g_camera.position
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    # Compute dot product to project each vector onto the ray
    ray_projection = (avatar_means - g_camera.position) @ ray_direction

    # Compute closest point on the ray for each head avatar mean
    closest_points_on_ray = g_camera.position + ray_projection[:, np.newaxis] * ray_direction

    # Compute distances between each head avatar mean and its closest point on the ray
    distances = np.linalg.norm(avatar_means - closest_points_on_ray, axis=1)

    # Get minimal distance 
    min_dist = np.min(distances)
    if min_dist >= DISPLACEMENT_FACTOR / 2:
        g_selected_head_avatar_index = -1
        g_selected_head_avatar_name = "None"
        return

    # Get index of the closest point
    closest_point_index = np.argmin(distances)

    # Get index and name of the selected head avatar
    g_selected_head_avatar_index = np.where(np.cumsum(g_head_avatar_checkboxes) - 1 == closest_point_index)[0][0]
    g_selected_head_avatar_name = "Head Avatar " + str(g_selected_head_avatar_index + 1)

def update_displacements_and_opacities():
    global gaussians

    for i in range(len(g_head_avatars)):
        xyz, _, _, opacity, _ = g_head_avatars[i].get_data()

        if g_head_avatar_checkboxes[i] and (g_show_hair[i] or g_show_head[i]):
            update_means(i)
            gaussians.opacity[i*N_GAUSSIANS:(i+1)*N_GAUSSIANS, :] = np.vstack([opacity[:N_HAIR_GAUSSIANS, :] * g_show_hair[i], opacity[N_HAIR_GAUSSIANS:, :] * g_show_head[i]])
        else:
            gaussians.opacity[i*N_GAUSSIANS:(i+1)*N_GAUSSIANS, :] = 0

def update_head_opacity():
    i = g_selected_head_avatar_index
    if g_show_head[i]:
        _, _, _, opacity, _ = g_head_avatars[i].get_data()
        gaussians.opacity[i*N_GAUSSIANS+N_HAIR_GAUSSIANS:(i+1)*N_GAUSSIANS, :] = opacity[N_HAIR_GAUSSIANS:, :]
    else:
        gaussians.opacity[i*N_GAUSSIANS+N_HAIR_GAUSSIANS:(i+1)*N_GAUSSIANS, :] = 0

def update_hair_opacity():
    i = g_selected_head_avatar_index
    if g_show_hair[i]:
        _, _, _, opacity, _ = g_head_avatars[i].get_data()
        gaussians.opacity[i*N_GAUSSIANS:i*N_GAUSSIANS+N_HAIR_GAUSSIANS, :] = opacity[:N_HAIR_GAUSSIANS, :]
    else:
        gaussians.opacity[i*N_GAUSSIANS:i*N_GAUSSIANS+N_HAIR_GAUSSIANS, :] = 0

def update_head_color():
    i = g_selected_head_avatar_index
    if g_show_head_color[i]:
        gaussians.sh[i*N_GAUSSIANS+N_HAIR_GAUSSIANS:(i+1)*N_GAUSSIANS, 0:3] = np.asarray(g_head_color[i]).T
    else:
        _, _, _, _, sh = g_head_avatars[i].get_data()
        gaussians.sh[i*N_GAUSSIANS+N_HAIR_GAUSSIANS:(i+1)*N_GAUSSIANS, :] = sh[N_HAIR_GAUSSIANS:, :]

def update_hair_color():
    i = g_selected_head_avatar_index
    if g_show_hair_color[i]:
        gaussians.sh[i*N_GAUSSIANS:i*N_GAUSSIANS+N_HAIR_GAUSSIANS, 0:3] = np.asarray(g_hair_color[i]).T
    else:
        _, _, _, _, sh = g_head_avatars[i].get_data()
        gaussians.sh[i*N_GAUSSIANS:i*N_GAUSSIANS+N_HAIR_GAUSSIANS, :] = sh[:N_HAIR_GAUSSIANS, :]

def update_hair_scale():
    i = g_selected_head_avatar_index
    _, _, scale, _, _ = g_head_avatars[i].get_data()
    gaussians.scale[i*N_GAUSSIANS:i*N_GAUSSIANS+N_HAIR_GAUSSIANS, :] = scale[:N_HAIR_GAUSSIANS] * g_hair_scale[i]

def update_frame():
    i = g_selected_head_avatar_index
    _, rot = get_frame(i)
    gaussians.rot[i*N_GAUSSIANS:i*N_GAUSSIANS+N_HAIR_GAUSSIANS, :] = rot
    update_means(i)

def update_means(head_avatar_index):
    i = head_avatar_index

    xyz, _, _, _, _ = g_head_avatars[i].get_data()
    gaussians.xyz[i*N_GAUSSIANS:(i+1)*N_GAUSSIANS, :] = xyz

    f, _ = get_frame(i)
    gaussians.xyz[i*N_GAUSSIANS:i*N_GAUSSIANS+N_HAIR_GAUSSIANS, :] = f

    d = get_displacement(i)
    gaussians.xyz[i*N_GAUSSIANS:(i+1)*N_GAUSSIANS, :] += d

    points = g_hair_points[i] + d
    c = get_curls(i)
    xyz, rot, scale = frenet_arcle.calculate_frenet_frame_t_npy(points+c)
    gaussians.xyz[i*N_GAUSSIANS:i*N_GAUSSIANS+N_HAIR_GAUSSIANS, :] = xyz.reshape(-1,3)
    gaussians.rot[i*N_GAUSSIANS:i*N_GAUSSIANS+N_HAIR_GAUSSIANS, :] = rot.reshape(-1,4)
    gaussians.scale[i*N_GAUSSIANS:i*N_GAUSSIANS+N_HAIR_GAUSSIANS, :] = scale.reshape(-1,3)

    g_head_avatar_means[i] = np.mean(gaussians.xyz[i*N_GAUSSIANS:(i+1)*N_GAUSSIANS], axis=0)

def get_displacement(head_avatar_index):
    i = head_avatar_index
    j = np.cumsum(g_head_avatar_checkboxes)[i] - 1
    return np.array([j * DISPLACEMENT_FACTOR, 0, 0]).astype(np.float32)

def get_hair_points(xyz, rot, scale):
    i = len(g_head_avatars)-1

    strands = [] # shape is 4000, 32, 3
    for k in range(N_HAIR_STRANDS):
        strand_xyz = xyz[k*N_GAUSSIANS_PER_STRAND:(k+1)*N_GAUSSIANS_PER_STRAND, :]
        strand_rot = rot[k*N_GAUSSIANS_PER_STRAND:(k+1)*N_GAUSSIANS_PER_STRAND, :]
        strand_scale = scale[k*N_GAUSSIANS_PER_STRAND:(k+1)*N_GAUSSIANS_PER_STRAND, :]

        r, x, y, z = strand_rot[0]
        displacement = 0.5*strand_scale[0]*np.array([1. - 2. * (y * y + z * z), 2. * (x * y + r * z), 2. * (x * z - r * y)])
        points = [strand_xyz[0]-displacement] # shape is 32, 3
        for j in range(N_GAUSSIANS_PER_STRAND):
            r, x, y, z = strand_rot[j]
            displacement = 0.5*strand_scale[j]*np.array([1. - 2. * (y * y + z * z), 2. * (x * y + r * z), 2. * (x * z - r * y)])
            points.append(strand_xyz[j]+displacement)
        strands.append(np.array(points))

    return np.array(strands)

def get_curls(head_avatar_index):
    i = head_avatar_index
    t = np.linspace(0, 1, N_GAUSSIANS_PER_STRAND+1)
    return ((2*t)**2*g_wave_amplitude[i] * np.sin(2 * np.pi * g_wave_frequency[i] * t))[:, np.newaxis].astype(np.float32)

def get_frame(head_avatar_index):
    i = head_avatar_index
    if g_frame[i] > 0:
        xyz = np.load(f"./models/head ({head_file})/320_to_320/frame_{g_frame[i]}_mean_frenet.npy").reshape(-1, 3)
        rot = np.load(f"./models/head ({head_file})/320_to_320/frame_{g_frame[i]}_rot_frenet.npy").reshape(-1, 3, 3)
        rot = np.array([rotmat2qvec(R) for R in rot])
        _, _, scale, _, _ = g_head_avatars[i].get_data()
        g_hair_points[i] = get_hair_points(xyz, rot, scale)
    else:
        xyz, rot, _, _, _ = g_head_avatars[i].get_data()
    return xyz[:N_HAIR_GAUSSIANS, :].astype(np.float32), rot[:N_HAIR_GAUSSIANS, :].astype(np.float32)

def cut_hair():
    # Get hair gaussians of selected head avatar
    i = g_selected_head_avatar_index
    hair_gaussians = gaussians.xyz[i*N_GAUSSIANS:i*N_GAUSSIANS+N_HAIR_GAUSSIANS, :]

    # Get mouse 3D position
    mouse_pos_2d = imgui.get_io().mouse_pos
    mouse_pos_3d = util.glhUnProjectf(mouse_pos_2d.x, mouse_pos_2d.y, 1, g_camera.get_view_matrix(), g_camera.get_project_matrix(), gl.glGetIntegerv(gl.GL_VIEWPORT))

    # Compute ray direction
    ray_direction = mouse_pos_3d - g_camera.position
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    # Compute dot product to project each vector onto the ray
    ray_projection = (hair_gaussians - g_camera.position) @ ray_direction

    # Compute closest point on the ray for each hair gaussian
    closest_points_on_ray = g_camera.position + ray_projection[:, np.newaxis] * ray_direction

    # Compute distances between each head avatar mean and its closest point on the ray
    distances = np.linalg.norm(hair_gaussians - closest_points_on_ray, axis=1)

    # Zero the opacity of the closest hair gaussians
    g_head_avatars[i].opacity[:N_HAIR_GAUSSIANS, :][distances < g_max_cutting_distance, :] = 0
    gaussians.opacity[i*N_GAUSSIANS:i*N_GAUSSIANS+N_HAIR_GAUSSIANS, :][distances < g_max_cutting_distance, :] = 0

    # Make sure there are no flying strands
    for j in range(N_HAIR_STRANDS):
        k = np.where(g_head_avatars[i].opacity[j*N_GAUSSIANS_PER_STRAND:(j+1)*N_GAUSSIANS_PER_STRAND, :] == 0)[0]
        if len(k) > 0:
            k = k[0]
            g_head_avatars[i].opacity[j*N_GAUSSIANS_PER_STRAND+k:(j+1)*N_GAUSSIANS_PER_STRAND, :] = 0
            gaussians.opacity[i*N_GAUSSIANS+j*N_GAUSSIANS_PER_STRAND+k:i*N_GAUSSIANS+(j+1)*N_GAUSSIANS_PER_STRAND, :] = 0

def reset_cut():
    file_path = f"./models/head ({head_file})/point_cloud/iteration_30000/point_cloud.ply"
    if file_path:
        try:
            # Set opacity from original head avatar
            _, _, _, opacity, _ = util_gau.load_ply(file_path).get_data()
            i = g_selected_head_avatar_index
            g_head_avatars[i].opacity[:N_HAIR_GAUSSIANS, :] = opacity[:N_HAIR_GAUSSIANS, :]
            update_hair_opacity()
        except RuntimeError as e:
                pass

#####################

def impl_glfw_init():
    window_name = "Dynamic Gaussian Visualizer"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    global window
    window = glfw.create_window(
        g_camera.w, g_camera.h, window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False
    if g_cutting_mode or g_render_mode == 0:
        g_renderer.update_ray_direction(g_camera, imgui.get_io().mouse_pos)
    g_camera.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)

    # Mouse Controller Variables
    global left_click_start_time, right_click_start_time

    # Record the time when the left mouse button is pressed
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        left_click_start_time = time.time()

    # Record the time when the right mouse button is pressed
    if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
        right_click_start_time = time.time()

    # Select closest head avatar
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
        end_time = time.time()
        left_click_duration = end_time - left_click_start_time
        
        if left_click_duration < CLICK_THRESHOLD:
            select_closest_head_avatar()
            g_renderer.update_selected_head_avatar_index(g_selected_head_avatar_index)

    # Cut
    if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.RELEASE:
        end_time = time.time()
        right_click_duration = end_time - right_click_start_time
        
        if right_click_duration < CLICK_THRESHOLD and g_cutting_mode and g_selected_head_avatar_index != -1:
            cut_hair()
            g_renderer.update_gaussian_data(gaussians)

def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)

def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)

def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:
        g_renderer.update_camera_pose(g_camera)
        g_camera.is_pose_dirty = False

def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera)
        g_camera.is_intrin_dirty = False

def update_activated_renderer_state(gaussians: util_gau.GaussianData):
    g_renderer.update_gaussian_data(gaussians)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)


def main():
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort, \
        g_show_control_win, g_show_help_win, g_show_camera_win, \
        g_render_mode, g_render_mode_tables

    # Head Avatars Global Variables
    global gaussians, g_show_head_avatars_win, g_head_avatar_checkboxes, g_empty_gaussian, g_cutting_mode, \
        g_coloring_mode, g_max_cutting_distance, g_z_min, g_z_max, g_z_plane, g_invert_z_plane

    # Head Avatar Controller Global Variables
    global g_show_head_avatar_controller_win, g_selected_head_avatar_index, g_selected_head_avatar_name, \
        g_show_hair, g_show_head, g_hair_color, g_head_color, g_show_hair_color, g_show_head_color, g_hair_scale, \
        g_wave_frequency, g_wave_amplitude, g_frame
        
    imgui.create_context()
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()
   
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    
    glfw.set_window_size_callback(window, window_resize_callback)

    # init renderer
    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)
    g_renderer_list[BACKEND_OGL_AXES] = OpenGLRendererAxes(g_camera.w, g_camera.h)
    try:
        from renderer_cuda import CUDARenderer
        g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
    except ImportError:
        g_renderer_idx = BACKEND_OGL
    else:
        g_renderer_idx = BACKEND_CUDA

    g_renderer = g_renderer_list[g_renderer_idx]

    # gaussian data
    update_activated_renderer_state(gaussians)    

    # Set parameters in shader
    g_renderer.update_N_HAIR_GAUSSIANS(N_HAIR_GAUSSIANS)
    g_renderer.update_cutting_mode(g_cutting_mode)
    g_renderer.update_selected_head_avatar_index(g_selected_head_avatar_index)
    g_renderer.update_max_cutting_distance(g_max_cutting_distance)
    g_renderer.update_coloring_mode(g_coloring_mode)
    g_renderer.update_invert_z_plane(g_invert_z_plane)
    g_renderer.update_z_plane(g_z_plane)

    # maximize window
    glfw.maximize_window(window)
    
    # settings
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose_lazy()
        update_camera_intrin_lazy()
        
        g_renderer.draw()

        # imgui ui
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_control_win = imgui.menu_item(
                    "Show Control", None, g_show_control_win
                )
                clicked, g_show_help_win = imgui.menu_item(
                    "Show Help", None, g_show_help_win
                )
                clicked, g_show_camera_win = imgui.menu_item(
                    "Show Camera Control", None, g_show_camera_win
                )
                clicked, g_show_head_avatars_win = imgui.menu_item(
                    "Show Head Avatars", None, g_show_head_avatars_win
                )
                clicked, g_show_head_avatar_controller_win = imgui.menu_item(
                    "Show Head Avatar Controller", None, g_show_head_avatar_controller_win
                )
                imgui.end_menu()
            imgui.end_main_menu_bar()
        
        if g_show_control_win:
            if imgui.begin("Control", True):
                # rendering backend
                changed, g_renderer_idx = imgui.combo("backend", g_renderer_idx, ["ogl", "cuda", "ogl_axes"][:len(g_renderer_list)])
                if changed:
                    g_renderer = g_renderer_list[g_renderer_idx]
                    update_activated_renderer_state(gaussians)

                imgui.text(f"fps = {imgui.get_io().framerate:.1f}")

                changed, g_renderer.reduce_updates = imgui.checkbox( "reduce updates", g_renderer.reduce_updates,)

                imgui.text(f"# of Gaus = {gaussians.xyz.shape[0]}")
                if imgui.button(label='open ply'):
                    file_path = filedialog.askopenfilename(title="open ply",
                        initialdir="./data",
                        filetypes=[('ply file', '.ply')]
                        )
                    if file_path:
                        try:
                            gaussians = util_gau.load_ply(file_path)
                            g_renderer.update_gaussian_data(gaussians)
                            g_renderer.sort_and_update(g_camera)
                        except RuntimeError as e:
                            pass
                
                # camera fov
                changed, g_camera.fovy = imgui.slider_float(
                    "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
                )
                g_camera.is_intrin_dirty = changed
                update_camera_intrin_lazy()
                
                # scale modifier
                changed, g_scale_modifier = imgui.slider_float(
                    "scale", g_scale_modifier, 0.1, 10, "scale modifier = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset"):
                    g_scale_modifier = 1.
                    changed = True
                    
                if changed:
                    g_renderer.set_scale_modifier(g_scale_modifier)
                
                # render mode
                changed, g_render_mode = imgui.combo("shading", g_render_mode, g_render_mode_tables)
                if changed:
                    g_renderer.set_render_mod(g_render_mode - 5)
                
                # sort button
                if imgui.button(label='sort Gaussians'):
                    g_renderer.sort_and_update(g_camera)
                imgui.same_line()
                changed, g_auto_sort = imgui.checkbox(
                        "auto sort", g_auto_sort,
                    )
                if g_auto_sort:
                    g_renderer.sort_and_update(g_camera)
                
                if imgui.button(label='save image'):
                    width, height = glfw.get_framebuffer_size(window)
                    nrChannels = 3;
                    stride = nrChannels * width;
                    stride += (4 - stride % 4) if stride % 4 else 0
                    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                    gl.glReadBuffer(gl.GL_FRONT)
                    bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                    img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
                    imageio.imwrite("save.png", img[::-1])
                    # save intermediate information
                    # np.savez(
                    #     "save.npz",
                    #     gau_xyz=gaussians.xyz,
                    #     gau_s=gaussians.scale,
                    #     gau_rot=gaussians.rot,
                    #     gau_c=gaussians.sh,
                    #     gau_a=gaussians.opacity,
                    #     viewmat=g_camera.get_view_matrix(),
                    #     projmat=g_camera.get_project_matrix(),
                    #     hfovxyfocal=g_camera.get_htanfovxy_focal()
                    # )
                imgui.end()

        if g_show_camera_win:
            if imgui.button(label='rot 180'):
                g_camera.flip_ground()

            changed, g_camera.target_dist = imgui.slider_float(
                    "t", g_camera.target_dist, 1., 8., "target dist = %.3f"
                )
            if changed:
                g_camera.update_target_distance()

            changed, g_camera.rot_sensitivity = imgui.slider_float(
                    "r", g_camera.rot_sensitivity, 0.002, 0.1, "rotate speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset r"):
                g_camera.rot_sensitivity = 0.002

            changed, g_camera.trans_sensitivity = imgui.slider_float(
                    "m", g_camera.trans_sensitivity, 0.001, 0.03, "move speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset m"):
                g_camera.trans_sensitivity = 0.01

            changed, g_camera.zoom_sensitivity = imgui.slider_float(
                    "z", g_camera.zoom_sensitivity, 0.001, 0.05, "zoom speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset z"):
                g_camera.zoom_sensitivity = 0.01

            changed, g_camera.roll_sensitivity = imgui.slider_float(
                    "ro", g_camera.roll_sensitivity, 0.003, 0.1, "roll speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset ro"):
                g_camera.roll_sensitivity = 0.03

        if g_show_help_win:
            imgui.begin("Help", True)
            imgui.text("Open Gaussian Splatting PLY file \n  by click 'open ply' button")
            imgui.text("Use left click & move to rotate camera")
            imgui.text("Use right click & move to translate camera")
            imgui.text("Press Q/E to roll camera")
            imgui.text("Use scroll to zoom in/out")
            imgui.text("Use control panel to change setting")
            imgui.end()

        # Head Avatars Window
        if g_show_head_avatars_win:
            imgui.begin("Head Avatars", True)

            # Load Head avatar button
            if imgui.button(label='open head avatar ply'):
                open_head_avatar_ply()
                update_displacements_and_opacities()
                g_renderer.update_gaussian_data(gaussians)
                
            # Display Head Avatar Checkboxes
            for i in range(len(g_head_avatars)):
                changed, g_head_avatar_checkboxes[i] = imgui.checkbox(f"Head Avatar {i + 1}", g_head_avatar_checkboxes[i])
                if changed:
                    update_displacements_and_opacities()
                    g_renderer.update_gaussian_data(gaussians)
            
            imgui.end()

        # Head Avatar Controller Window
        if g_show_head_avatar_controller_win:
            imgui.begin("Head Avatar Controller", True)

            imgui.text(f"Selected: {g_selected_head_avatar_name}")

            if g_selected_head_avatar_index != -1:
                i = g_selected_head_avatar_index

                changed, g_cutting_mode = imgui.checkbox("Cutting Mode", g_cutting_mode)
                if changed:
                    g_renderer.update_cutting_mode(g_cutting_mode)

                changed, g_max_cutting_distance = imgui.slider_float("Cutting Area", g_max_cutting_distance, 0.01, 0.5, "%.2f")
                if changed:
                    g_renderer.update_max_cutting_distance(g_max_cutting_distance)

                if imgui.button(label="Reset Hair Style"):
                    reset_cut()
                    g_renderer.update_gaussian_data(gaussians)

                changed, g_coloring_mode = imgui.checkbox("Coloring Mode", g_coloring_mode)
                if changed:
                    g_renderer.update_coloring_mode(g_coloring_mode)

                changed, g_z_plane = imgui.slider_float("Z-Plane", g_z_plane, g_z_min, g_z_max, "z = %.3f")
                if changed:
                    g_renderer.update_z_plane(g_z_plane)

                imgui.same_line()

                changed, g_invert_z_plane = imgui.checkbox("Invert Z-Plane", g_invert_z_plane)
                if changed:
                    g_renderer.update_invert_z_plane(g_invert_z_plane)

                changed, g_show_hair[i] = imgui.checkbox("Show Hair", g_show_hair[i])
                if changed:
                    update_hair_opacity()
                    g_renderer.update_gaussian_data(gaussians)

                imgui.same_line()

                changed, g_show_head[i] = imgui.checkbox("Show Head", g_show_head[i])
                if changed:
                    update_head_opacity()
                    g_renderer.update_gaussian_data(gaussians)

                changed, g_hair_color[i] = imgui.color_edit3("Hair Color", *g_hair_color[i])
                if changed:
                    update_hair_color()
                    g_renderer.update_gaussian_data(gaussians)

                imgui.same_line()

                changed, g_show_hair_color[i] = imgui.checkbox("Show Hair Color", g_show_hair_color[i])
                if changed:
                    update_hair_color()
                    g_renderer.update_gaussian_data(gaussians)

                changed, g_head_color[i] = imgui.color_edit3("Head Color", *g_head_color[i])
                if changed:
                    update_head_color()
                    g_renderer.update_gaussian_data(gaussians)

                imgui.same_line()

                changed, g_show_head_color[i] = imgui.checkbox("Show Head Color", g_show_head_color[i])
                if changed:
                    update_head_color()
                    g_renderer.update_gaussian_data(gaussians)

                changed, g_hair_scale[i] = imgui.slider_float("Hair Scale", g_hair_scale[i], 0.5, 2, "Hair Scale = %.3f")
                if changed:
                    update_hair_scale()
                    g_renderer.update_gaussian_data(gaussians)

                imgui.same_line()

                if imgui.button(label="Reset Hair Scale"):
                    g_hair_scale[i] = 1
                    update_hair_scale()
                    g_renderer.update_gaussian_data(gaussians)   

                changed, g_wave_frequency[i] = imgui.slider_float("Wave Frequency", g_wave_frequency[i], 0, 5, "Wave Frequency = %.2f")
                if changed:
                    update_means(g_selected_head_avatar_index)
                    g_renderer.update_gaussian_data(gaussians)

                imgui.same_line()

                if imgui.button(label="Reset Wave Frequency"):
                    g_wave_frequency[i] = 0
                    update_means(g_selected_head_avatar_index)
                    g_renderer.update_gaussian_data(gaussians) 

                changed, g_wave_amplitude[i] = imgui.slider_float("Wave Amplitude", g_wave_amplitude[i], 0, 0.05, "Wave Amplitude = %.3f")
                if changed:
                    update_means(g_selected_head_avatar_index)
                    g_renderer.update_gaussian_data(gaussians)

                imgui.same_line()

                if imgui.button(label="Reset Wave Height"):
                    g_wave_amplitude[i] = 0
                    update_means(g_selected_head_avatar_index)
                    g_renderer.update_gaussian_data(gaussians) 

                changed, g_frame[i] = imgui.slider_int("Frame", g_frame[i], 0, 98, "Frame = %d")
                if changed:
                    update_frame()
                    g_renderer.update_gaussian_data(gaussians)     

            imgui.end()
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description="Dynamic Gaussian Visualizer")
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    args = parser.parse_args()

    main()
