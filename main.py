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

N_HAIR_GAUSSIANS = 4650
N_HAIR_STRANDS = 150
N_GAUSSIANS_PER_STRAND = 31

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
g_frame_modifier = 1
g_show_input_init = False
g_show_random_init = False
g_auto_sort = False
g_show_control_win = True
g_show_help_win = True
g_show_camera_win = True
g_show_head_avatar_win = True
g_use_hair_color = False
g_use_head_color = False
g_show_hair = True
g_show_head = True
g_hair_color = np.asarray([0, 1, 0])
g_head_color = np.asarray([0, 0, 1])
g_color_strands = False
g_render_mode_tables = ["Gaussian Ball", "Flat Ball", "Billboard", "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 7

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
    render(gaussians)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_frame_modifier(np.interp(g_frame_modifier, [1, 300], [100, 1]))
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)

def render(gaussians):
    g_renderer.update_gaussian_data(color(part(gaussians)))

def part(gaussians):
    if g_show_hair and g_show_head:
        return gaussians
    elif g_show_hair:
        return util_gau.GaussianData(
            gaussians.xyz[:N_HAIR_GAUSSIANS, :],
            gaussians.rot[:N_HAIR_GAUSSIANS, :],
            gaussians.scale[:N_HAIR_GAUSSIANS, :],
            gaussians.opacity[:N_HAIR_GAUSSIANS, :],
            gaussians.sh[:N_HAIR_GAUSSIANS, :],
        )
    elif g_show_head:
        return util_gau.GaussianData(
            gaussians.xyz[N_HAIR_GAUSSIANS:, :],
            gaussians.rot[N_HAIR_GAUSSIANS:, :],
            gaussians.scale[N_HAIR_GAUSSIANS:, :],
            gaussians.opacity[N_HAIR_GAUSSIANS:, :],
            gaussians.sh[N_HAIR_GAUSSIANS:, :],
        )
    else:
        return util_gau.GaussianData(
            gaussians.xyz[:0, :],
            gaussians.rot[:0, :],
            gaussians.scale[:0, :],
            gaussians.opacity[:0, :],
            gaussians.sh[:0, :],
        )

def color(gaussians):
    if not g_use_hair_color and not g_use_head_color and not g_color_strands:
        return gaussians
    
    colors = gaussians.sh[:, :3].copy()
    if g_use_hair_color:
        colors[:N_HAIR_GAUSSIANS, :] = g_hair_color
    if g_use_head_color:
        colors[N_HAIR_GAUSSIANS:, :] = g_head_color
    if g_color_strands:
        for i in range(N_HAIR_STRANDS):
            colors[i*N_GAUSSIANS_PER_STRAND:(i+1)*N_GAUSSIANS_PER_STRAND, :] = np.random.rand(1, 3)

    return util_gau.GaussianData(
            gaussians.xyz,
            gaussians.rot,
            gaussians.scale,
            gaussians.opacity,
            colors.astype(np.float32),
        )  


def main():
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_frame_modifier, g_show_input_init, g_show_random_init, g_auto_sort, g_show_hair, g_show_head, g_hair_color, g_head_color, g_use_hair_color, g_use_head_color, g_color_strands, \
        g_show_control_win, g_show_help_win, g_show_camera_win, g_show_head_avatar_win, \
        g_render_mode, g_render_mode_tables
        
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
    input_gaussians = None
    gaussians = util_gau.naive_gaussian()
    random_gaussians = util_gau.random_gaussian(gaussians)
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
                clicked, g_show_head_avatar_win = imgui.menu_item(
                    "Show Head Avatar Control", None, g_show_head_avatar_win
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

                imgui.text(f"# of Gaus = {len(gaussians)}")
                if imgui.button(label='open ply'):
                    file_path = filedialog.askopenfilename(title="open ply",
                        initialdir="D:\\Daniel\\Masters\\Term 2\\Practical Machine Learning\\Models",
                        filetypes=[('ply file', '.ply')]
                        )
                    if file_path:
                        try:
                            gaussians = util_gau.load_ply(file_path)
                            random_gaussians = util_gau.random_gaussian(gaussians)
                            g_show_input_init = False
                            g_show_random_init = False
                            render(gaussians)
                            g_renderer.sort_and_update(g_camera)
                        except RuntimeError as e:
                            pass

                if imgui.button(label='open input ply'):
                    file_path = filedialog.askopenfilename(title="open input ply",
                        initialdir="D:\\Daniel\\Masters\\Term 2\\Practical Machine Learning\\Models",
                        filetypes=[('ply file', '.ply')]
                        )
                    if file_path:
                        try:
                            input_gaussians = util_gau.load_input_ply(file_path)
                            g_show_input_init = True
                            g_show_random_init = False
                            g_frame_modifier = 300
                            g_renderer.set_frame_modifier(np.interp(g_frame_modifier, [1, 300], [100, 1]))
                            g_renderer.update_gaussian_data(input_gaussians)
                            g_renderer.sort_and_update(g_camera)                            
                        except RuntimeError as e:
                            pass

                changed, g_show_input_init = imgui.checkbox( "show intput init", g_show_input_init,)

                if changed:
                    if g_show_input_init:
                        g_show_random_init = False
                        if input_gaussians is not None:
                            g_renderer.update_gaussian_data(input_gaussians)
                    else:
                        render(gaussians)

                changed, g_show_random_init = imgui.checkbox( "show random init", g_show_random_init,)

                if changed:
                    if g_show_random_init:
                        g_show_input_init = False
                        g_renderer.update_gaussian_data(random_gaussians)
                    else:
                        render(gaussians)
                
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

                # frame modifier
                changed, g_frame_modifier = imgui.slider_int(
                    "frame", g_frame_modifier, 1, 300, "Frames = %d"
                )

                if changed:
                    g_renderer.set_frame_modifier(np.interp(g_frame_modifier, [1, 300], [100, 1]))
                
                # render mode
                changed, g_render_mode = imgui.combo("shading", g_render_mode, g_render_mode_tables)
                if changed:
                    g_renderer.set_render_mod(g_render_mode - 4)
                
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
                    "m", g_camera.trans_sensitivity, 0.001, 2, "move speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset m"):
                g_camera.trans_sensitivity = 0.01

            changed, g_camera.zoom_sensitivity = imgui.slider_float(
                    "z", g_camera.zoom_sensitivity, 0.001, 2, "zoom speed = %.3f"
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

        if g_show_head_avatar_win:
            imgui.begin("Head Avatar Control", True)
            
            if imgui.button(label="Re-center Head Avatar"):
                g_camera.position = np.array([0.0, 0.5, 2]).astype(np.float32)
                g_camera.target = np.array([0.0, 0.4, 0.3]).astype(np.float32)
                g_camera.up = np.array([0.0, 1.0, 0.0]).astype(np.float32)
                g_camera.pitch = -0.184
                g_camera.is_pose_dirty = True

            changed, g_show_hair = imgui.checkbox("Hair", g_show_hair)
            if changed:
                render(gaussians)

            changed, g_show_head = imgui.checkbox("Head", g_show_head)
            if changed:
                render(gaussians)

            changed, g_hair_color = imgui.color_edit3("Hair Color", *g_hair_color)
            if (changed):
                render(gaussians)

            imgui.same_line()

            changed, g_use_hair_color = imgui.checkbox("Use Hair Color", g_use_hair_color)
            if changed:
                g_color_strands = False
                render(gaussians)
            
            changed, g_head_color = imgui.color_edit3("Head Color", *g_head_color)
            if (changed):
                render(gaussians)

            imgui.same_line()

            changed, g_use_head_color = imgui.checkbox("Use Head Color", g_use_head_color)
            if changed:
                render(gaussians)

            changed, g_color_strands = imgui.checkbox("Color Strands", g_color_strands)
            if changed:
                g_use_hair_color = False
                render(gaussians)

            imgui.end()

        if g_show_help_win:
            imgui.begin("Help", True)
            imgui.text("Open Gaussian Splatting PLY file \n  by click 'open ply' button")
            imgui.text("Use left click & move to rotate camera")
            imgui.text("Use right click & move to translate camera")
            imgui.text("Press Q/E to roll camera")
            imgui.text("Use scroll to zoom in/out")
            imgui.text("Use control panel to change setting")
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
