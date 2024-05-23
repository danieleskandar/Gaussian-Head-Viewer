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
    None, None, None # ogl
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

N_HAIR_GAUSSIANS = 4650
N_HAIR_STRANDS = 150
N_GAUSSIANS_PER_STRAND = 31

########################
# Head Avatars Variables
########################
gaussians = util_gau.naive_gaussian()
g_show_head_avatars_win = True
g_head_avatar_checkboxes = []
g_head_avatars = []
g_empty_gaussian = util_gau.GaussianData(np.empty((1, 3)), np.empty((1, 4)), np.empty((1, 3)), np.empty((1, 3)), np.empty((1, 3)))

######################
# Head Avatars Actions
######################
def open_head_avatar_ply():
    file_path = filedialog.askopenfilename(
        title="open ply",
        initialdir="D:\\Daniel\\Masters\\Term 2\\Practical Machine Learning\\Models\\hair\\point_cloud\\iteration_30000",
        filetypes=[('ply file', '.ply')]
    )
    if file_path:
        try:
            g_head_avatar = util_gau.load_ply(file_path)
            g_head_avatars.append(g_head_avatar)
            g_head_avatar_checkboxes.append(True)
            g_show_hair.append(True)
            g_show_head.append(True)
            g_hair_color.append([1, 0, 0])
            g_head_color.append([1, 1, 1])
            g_show_hair_color.append(False)
            g_show_head_color.append(False)
            g_hair_scale.append(1)
            g_wave_frequency.append(0)
            g_wave_height.append(0)
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
g_wave_height = []

################################
# Head Avatar Controller Actions
################################
def select_closest_head_avatar():
    global g_selected_head_avatar_index, g_selected_head_avatar_name

    if len(g_head_avatars) == 0 or np.sum(g_head_avatar_checkboxes) == 0:
        g_selected_head_avatar_index = -1
        g_selected_head_avatar_name = "None"
        return

    # Get merged points
    j = 0
    l = np.sum(g_head_avatar_checkboxes)
    all_xyz = []
    for i in range(len(g_head_avatars)):
        if g_head_avatar_checkboxes[i]:
            N = g_head_avatars[i].xyz.shape[0]
            d = np.hstack([np.ones((N, 1)) * j, np.zeros((N, 2))])
            xyz = g_head_avatars[i].xyz + d

            t = np.linspace(0, 1, N_GAUSSIANS_PER_STRAND)
            for k in range(N_HAIR_STRANDS):
                offset = g_wave_height[i] * np.sin(2 * np.pi * g_wave_frequency[i] * t)
                xyz[k*N_GAUSSIANS_PER_STRAND:(k+1)*N_GAUSSIANS_PER_STRAND, :] += offset[:, np.newaxis]

            all_xyz.append(xyz)

            j += 1
    all_xyz = np.vstack(all_xyz).astype(np.float32)

    # Get mouse 3D position
    mouse_pos_2d = imgui.get_io().mouse_pos
    mouse_pos_3d = util.glhUnProjectf(mouse_pos_2d.x, mouse_pos_2d.y, 1, g_camera.get_view_matrix(), g_camera.get_project_matrix(), gl.glGetIntegerv(gl.GL_VIEWPORT))
    # Compute ray direction
    ray_direction = mouse_pos_3d - g_camera.position
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    # Compute dot product to project each vector onto the ray
    ray_projection = (all_xyz - g_camera.position) @ ray_direction
    # Compute closest point on the ray for each point
    closest_points_on_ray = g_camera.position + ray_projection[:, np.newaxis] * ray_direction
    # Compute distances between each point and its closest point on the ray
    distances = np.linalg.norm(all_xyz - closest_points_on_ray, axis=1)

    # Get index of the closest point
    closest_point_index = np.argmin(distances)

    # Get index and name of the selected head avatar
    g_selected_head_avatar_index = np.where(np.cumsum(g_head_avatar_checkboxes) == closest_point_index // N + 1)[0][0]
    g_selected_head_avatar_name = "Head Avatar " + str(g_selected_head_avatar_index + 1)

#################
# 
#################
def merge_head_avatars():
    global g_selected_head_avatar_index, g_selected_head_avatar_name

    if len(g_head_avatars) == 0 or np.sum(g_head_avatar_checkboxes) == 0:
        g_selected_head_avatar_index = -1
        g_selected_head_avatar_name = "None"
        return g_empty_gaussian

    # N: number of gaussians in head avatar
    # j: counter used to calculate amount of displacement
    # l: number of selected head avatars
    # d: displacement added to the x component of the gaussian means
    j = 0
    l = np.sum(g_head_avatar_checkboxes)
    all_xyz = []
    all_rot = []
    all_scale = []
    all_opacity = []
    all_sh = []
    for i in range(len(g_head_avatars)):
        if g_head_avatar_checkboxes[i] and (g_show_hair[i] or g_show_head[i]):
            # Get data
            xyz, rot, scale, opacity, sh = g_head_avatars[i].get_data()

            # Add displacement to gaussian means
            N = g_head_avatars[i].xyz.shape[0]
            d = np.hstack([np.ones((N, 1)) * j, np.zeros((N, 2))])
            xyz = xyz + d

            # Color Data
            if g_show_hair[i] and g_show_hair_color[i]:
                sh[:N_HAIR_GAUSSIANS, 0:3] = np.asarray(g_hair_color[i]).T
            if g_show_head[i] and g_show_head_color[i]:
                sh[N_HAIR_GAUSSIANS:, 0:3] = np.asarray(g_head_color[i]).T

            # Scale Hair Data
            if g_show_hair[i]:
                scale[:N_HAIR_GAUSSIANS, :] *= g_hair_scale[i]

            # Apply curls and waves
            t = np.linspace(0, 1, N_GAUSSIANS_PER_STRAND)
            offset = g_wave_height[i] * np.sin(2 * np.pi * g_wave_frequency[i] * t)
            for k in range(N_HAIR_STRANDS):
                xyz[k*N_GAUSSIANS_PER_STRAND:(k+1)*N_GAUSSIANS_PER_STRAND, :] += offset[:, np.newaxis]
            
            # Slice data
            if g_show_hair[i] and not g_show_head[i]:
                xyz, rot, scale, opacity, sh = util_gau.slice_data(0, N_HAIR_GAUSSIANS, (xyz, rot, scale, opacity, sh))
            elif not g_show_hair[i] and g_show_head[i]:
                xyz, rot, scale, opacity, sh = util_gau.slice_data(N_HAIR_GAUSSIANS, N, (xyz, rot, scale, opacity, sh))

            all_xyz.append(xyz)
            all_rot.append(rot)
            all_scale.append(scale)
            all_opacity.append(opacity)
            all_sh.append(sh[:, 0:3])

            j += 1

    if len(all_xyz) == 0:
        g_selected_head_avatar_index = -1
        g_selected_head_avatar_name = "None"
        return g_empty_gaussian

    return util_gau.GaussianData(
        np.vstack(all_xyz).astype(np.float32),
        np.vstack(all_rot).astype(np.float32),
        np.vstack(all_scale).astype(np.float32),
        np.vstack(all_opacity).astype(np.float32),
        np.vstack(all_sh).astype(np.float32),
    )

def render_head_avatars():
    global gaussians
    gaussians = merge_head_avatars()
    g_renderer.update_gaussian_data(gaussians)

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
    g_camera.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)

    # Select closest head avatar
    if action == glfw.RELEASE:
        select_closest_head_avatar()

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
    global gaussians, g_show_head_avatars_win, g_head_avatar_checkboxes, g_empty_gaussian

    # # Head Avatar Controller Global Variables
    global g_show_head_avatar_controller_win, g_selected_head_avatar_index, g_selected_head_avatar_name, \
        g_show_hair, g_show_head, g_hair_color, g_head_color, g_show_hair_color, g_show_head_color, g_hair_scale, \
        g_wave_frequency, g_wave_height
        
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
        if (g_renderer_idx != 2):
            g_renderer.update_ray_direction(g_camera, imgui.get_io().mouse_pos)
        
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
                        initialdir="D:\\Daniel\\Masters\\Term 2\\Practical Machine Learning\\Models",
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
                render_head_avatars()
                
            # Display Head Avatar Checkboxes
            for i in range(len(g_head_avatars)):
                changed, g_head_avatar_checkboxes[i] = imgui.checkbox(f"Head Avatar {i + 1}", g_head_avatar_checkboxes[i])
                if changed:
                    render_head_avatars()
            
            imgui.end()

        # Head Avatar Controller Window
        if g_show_head_avatar_controller_win:
            imgui.begin("Head Avatar Controller", True)

            imgui.text(f"Selected: {g_selected_head_avatar_name}")

            if g_selected_head_avatar_index != -1:
                i = g_selected_head_avatar_index

                changed, g_show_hair[i] = imgui.checkbox("Show Hair", g_show_hair[i])
                if changed:
                    render_head_avatars()

                imgui.same_line()

                changed, g_show_head[i] = imgui.checkbox("Show Head", g_show_head[i])
                if changed:
                    render_head_avatars()

                changed, g_hair_color[i] = imgui.color_edit3("Hair Color", *g_hair_color[i])
                if changed:
                    render_head_avatars()

                imgui.same_line()

                changed, g_show_hair_color[i] = imgui.checkbox("Show Hair Color", g_show_hair_color[i])
                if changed:
                    render_head_avatars()

                changed, g_head_color[i] = imgui.color_edit3("Head Color", *g_head_color[i])
                if changed:
                    render_head_avatars()

                imgui.same_line()

                changed, g_show_head_color[i] = imgui.checkbox("Show Head Color", g_show_head_color[i])
                if changed:
                    render_head_avatars()

                changed, g_hair_scale[i] = imgui.slider_float("Hair Scale", g_hair_scale[i], 0.5, 2, "Hair Scale = %.3f")
                if changed:
                    render_head_avatars()

                imgui.same_line()

                if imgui.button(label="Reset Hair Scale"):
                    g_hair_scale[i] = 1
                    render_head_avatars()   

                changed, g_wave_frequency[i] = imgui.slider_float("Wave Frequency", g_wave_frequency[i], 0, 5, "Wave Frequency = %.2f")
                if changed:
                    render_head_avatars()

                imgui.same_line()

                if imgui.button(label="Reset Wave Frequency"):
                    g_wave_frequency[i] = 0
                    render_head_avatars() 

                changed, g_wave_height[i] = imgui.slider_float("Wave Height", g_wave_height[i], 0, 0.05, "Wave Height = %.3f")
                if changed:
                    render_head_avatars()

                imgui.same_line()

                if imgui.button(label="Reset Wave Height"):
                    g_wave_height[i] = 0
                    render_head_avatars()      

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
