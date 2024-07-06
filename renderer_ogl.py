from OpenGL import GL as gl
import util
import util_gau
import numpy as np

try:
    from OpenGL.raw.WGL.EXT.swap_control import wglSwapIntervalEXT
except:
    wglSwapIntervalEXT = None


_sort_buffer_xyz = None
_sort_buffer_gausid = None  # used to tell whether gaussian is reloaded

def _sort_gaussian_cpu(gaus, view_mat):
    xyz = np.asarray(gaus.xyz)
    view_mat = np.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = np.argsort(depth)
    index = index.astype(np.int32).reshape(-1, 1)
    return index


def _sort_gaussian_cupy(gaus, view_mat):
    import cupy as cp
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = cp.asarray(gaus.xyz)
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = cp.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = cp.argsort(depth)
    index = index.astype(cp.int32).reshape(-1, 1)

    index = cp.asnumpy(index) # convert to numpy
    return index


def _sort_gaussian_torch(gaus, view_mat):
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = torch.tensor(gaus.xyz).cuda()
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = torch.tensor(view_mat).cuda()
    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    index = torch.argsort(depth)
    index = index.type(torch.int32).reshape(-1, 1).cpu().numpy()
    return index


# Decide which sort to use
_sort_gaussian = None
try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
    print("Detect torch cuda installed, will use torch as sorting backend")
    _sort_gaussian = _sort_gaussian_torch
except ImportError:
    try:
        import cupy as cp
        print("Detect cupy installed, will use cupy as sorting backend")
        _sort_gaussian = _sort_gaussian_cupy
    except ImportError:
        _sort_gaussian = _sort_gaussian_cpu


class GaussianRenderBase:
    def __init__(self):
        self.gaussians = None
        self._reduce_updates = True

    @property
    def reduce_updates(self):
        return self._reduce_updates

    @reduce_updates.setter
    def reduce_updates(self, val):
        self._reduce_updates = val
        self.update_vsync()

    def update_vsync(self):
        return
        print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        raise NotImplementedError()
    
    def sort_and_update(self):
        raise NotImplementedError()

    def set_scale_modifier(self, modifier: float):
        raise NotImplementedError()
    
    def set_render_mod(self, mod: int):
        raise NotImplementedError()
    
    def update_camera_pose(self, camera: util.Camera):
        raise NotImplementedError()

    def update_camera_intrin(self, camera: util.Camera):
        raise NotImplementedError()

    def update_start(self, start):
        raise NotImplementedError()

    def update_n_gaussians(self, n_gaussians):
        raise NotImplementedError()

    def update_n_hair_gaussians(self, n_hair_gaussians):
        raise NotImplementedError()

    def update_cutting_mode(self, cutting_mode):
        raise NotImplementedError()

    def update_coloring_mode(self, coloring_mode):
        raise NotImplementedError()

    def update_keep_sh(self, keep_sh):
        raise NotImplementedError()

    def update_selected_color(self, selected_color):
        raise NotImplementedError()

    def update_max_coloring_distance(self, max_coloring_distance):
        raise NotImplementedError()

    def update_invert_x_plane(self, invert_x_plane):
        raise NotImplementedError()

    def update_invert_y_plane(self, invert_y_plane):
        raise NotImplementedError()

    def update_invert_z_plane(self, invert_z_plane):
        raise NotImplementedError()

    def update_selected_head_avatar_index(self, selected_head_avatar_index):
        raise NotImplementedError()

    def update_max_cutting_distance(self, max_cutting_distance):
        raise NotImplementedError()

    def update_x_plane(self, x_plane):
        raise NotImplementedError()

    def update_y_plane(self, y_plane):
        raise NotImplementedError()

    def update_z_plane(self, z_plane):
        raise NotImplementedError()

    def update_ray_direction(self, camera: util.Camera, mouse_pos_2d):
        raise NotImplementedError()
    
    def draw(self):
        raise NotImplementedError()
    
    def set_render_reso(self, w, h):
        raise NotImplementedError()


class OpenGLRenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        gl.glViewport(0, 0, w, h)
        self.program = util.load_shaders('shaders/gau_vert.glsl', 'shaders/gau_frag.glsl')

        # Vertex data for a quad
        self.quad_v = np.array([
            -1,  1,
            1,  1,
            1, -1,
            -1, -1
        ], dtype=np.float32).reshape(4, 2)
        self.quad_f = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32).reshape(2, 3)
        
        # load quad geometry
        vao, buffer_id = util.set_attributes(self.program, ["position"], [self.quad_v])
        util.set_faces_tovao(vao, self.quad_f)
        self.vao = vao
        self.gau_bufferid = None
        self.index_bufferid = None
        # opengl settings
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.update_vsync()

    def update_vsync(self):
        if wglSwapIntervalEXT is not None:
            wglSwapIntervalEXT(1 if self.reduce_updates else 0)
        else:
            return
            print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.gaussians = gaus
        # load gaussian geometry
        gaussian_data = gaus.flat()
        self.gau_bufferid = util.set_storage_buffer_data(self.program, "gaussian_data", gaussian_data, 
                                                         bind_idx=0,
                                                         buffer_id=self.gau_bufferid)
        util.set_uniform_1int(self.program, gaus.sh_dim, "sh_dim")

    def sort_and_update(self, camera: util.Camera):
        index = _sort_gaussian(self.gaussians, camera.get_view_matrix())
        self.index_bufferid = util.set_storage_buffer_data(self.program, "gi", index, 
                                                           bind_idx=1,
                                                           buffer_id=self.index_bufferid)
        return
   
    def set_scale_modifier(self, modifier):
        util.set_uniform_1f(self.program, modifier, "scale_modifier")

    def set_render_mod(self, mod: int):
        util.set_uniform_1int(self.program, mod, "render_mod")

    def set_render_reso(self, w, h):
        gl.glViewport(0, 0, w, h)

    def update_camera_pose(self, camera: util.Camera):
        view_mat = camera.get_view_matrix()
        util.set_uniform_mat4(self.program, view_mat, "view_matrix")
        util.set_uniform_v3(self.program, camera.position, "cam_pos")

    def update_camera_intrin(self, camera: util.Camera):
        proj_mat = camera.get_project_matrix()
        util.set_uniform_mat4(self.program, proj_mat, "projection_matrix")
        util.set_uniform_v3(self.program, camera.get_htanfovxy_focal(), "hfovxy_focal")

    def update_start(self, start):
        util.set_uniform_1int(self.program, start, "start_index")

    def update_n_gaussians(self, n_gaussians):
        util.set_uniform_1int(self.program, n_gaussians, "n_gaussians")

    def update_n_hair_gaussians(self, n_hair_gaussians):
        util.set_uniform_1int(self.program, n_hair_gaussians, "n_hair_gaussians")

    def update_cutting_mode(self, cutting_mode):
        util.set_uniform_1int(self.program, int(cutting_mode), "cutting_mode")

    def update_coloring_mode(self, coloring_mode):
        util.set_uniform_1int(self.program, int(coloring_mode), "coloring_mode")

    def update_keep_sh(self, keep_sh):
        util.set_uniform_1int(self.program, int(keep_sh), "keep_sh")

    def update_selected_color(self, selected_color):
        util.set_uniform_v3(self.program, selected_color, "selected_color")

    def update_max_coloring_distance(self, max_coloring_distance):
        util.set_uniform_1f(self.program, max_coloring_distance, "max_coloring_distance")

    def update_invert_x_plane(self, invert_x_plane):
        util.set_uniform_1int(self.program, int(invert_x_plane), "invert_x_plane")

    def update_invert_y_plane(self, invert_y_plane):
        util.set_uniform_1int(self.program, int(invert_y_plane), "invert_y_plane")
    
    def update_invert_z_plane(self, invert_z_plane):
        util.set_uniform_1int(self.program, int(invert_z_plane), "invert_z_plane")

    def update_selected_head_avatar_index(self, selected_head_avatar_index):
        util.set_uniform_1int(self.program, selected_head_avatar_index, "selected_head_avatar_index")

    def update_max_cutting_distance(self, max_cutting_distance):
        util.set_uniform_1f(self.program, max_cutting_distance, "max_cutting_distance")

    def update_x_plane(self, x_plane):
        util.set_uniform_1f(self.program, x_plane, "x_plane")

    def update_y_plane(self, y_plane):
        util.set_uniform_1f(self.program, y_plane, "y_plane")

    def update_z_plane(self, z_plane):
        util.set_uniform_1f(self.program, z_plane, "z_plane")

    def update_ray_direction(self, camera: util.Camera, mouse_pos_2d):
        mouse_pos_3d = util.glhUnProjectf(mouse_pos_2d.x, mouse_pos_2d.y, 1, camera.get_view_matrix(), camera.get_project_matrix(), gl.glGetIntegerv(gl.GL_VIEWPORT))
        ray_direction = mouse_pos_3d-camera.position
        ray_direction = ray_direction/np.linalg.norm(ray_direction)
        util.set_uniform_v3(self.program, ray_direction, "ray_direction")
   
    def draw(self):
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)
        num_gau = len(self.gaussians)
        # an instance renders 2 TRIANGLES, by rendering 6 different points, done as many times as number of gaussians
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(self.quad_f.reshape(-1)), gl.GL_UNSIGNED_INT, None, num_gau)

class OpenGLRendererAxes(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        gl.glViewport(0, 0, w, h)
        self.program = util.load_shaders('shaders/line_vert.glsl', 'shaders/line_frag.glsl')

        # Line data for gaussian axes
        self.lines = np.array([
            1,  0, 0,
            -1, 0, 0,
            0, 1, 0,
            0, -1, 0,
            0, 0, 1,
            0, 0, -1
        ], dtype=np.float32).reshape(6, 3)

        # load quad geometry
        vao, buffer_id = util.set_attributes(self.program, ["lines"], [self.lines])
        self.vao = vao
        self.gau_bufferid = None
        self.index_bufferid = None
        # opengl settings
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.update_vsync()

    def update_vsync(self):
        if wglSwapIntervalEXT is not None:
            wglSwapIntervalEXT(1 if self.reduce_updates else 0)

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.gaussians = gaus
        # load gaussian geometry
        gaussian_data = gaus.flat()
        self.gau_bufferid = util.set_storage_buffer_data(self.program, "gaussian_data", gaussian_data, 
                                                         bind_idx=0,
                                                         buffer_id=self.gau_bufferid)
        util.set_uniform_1int(self.program, gaus.sh_dim, "sh_dim")

    def sort_and_update(self, camera: util.Camera):
        index = _sort_gaussian(self.gaussians, camera.get_view_matrix())
        self.index_bufferid = util.set_storage_buffer_data(self.program, "gi", index, 
                                                           bind_idx=1,
                                                           buffer_id=self.index_bufferid)
        return
   
    def set_scale_modifier(self, modifier):
        util.set_uniform_1f(self.program, modifier, "scale_modifier")

    def set_render_mod(self, mod: int):
        util.set_uniform_1int(self.program, mod, "render_mod")

    def set_render_reso(self, w, h):
        gl.glViewport(0, 0, w, h)

    def update_camera_pose(self, camera: util.Camera):
        view_mat = camera.get_view_matrix()
        util.set_uniform_mat4(self.program, view_mat, "view_matrix")
        util.set_uniform_v3(self.program, camera.position, "cam_pos")

    def update_camera_intrin(self, camera: util.Camera):
        proj_mat = camera.get_project_matrix()
        util.set_uniform_mat4(self.program, proj_mat, "projection_matrix")
        util.set_uniform_v3(self.program, camera.get_htanfovxy_focal(), "hfovxy_focal")

    def update_start(self, start):
        util.set_uniform_1int(self.program, start, "start_index")

    def update_n_gaussians(self, n_gaussians):
        util.set_uniform_1int(self.program, n_gaussians, "n_gaussians")

    def update_n_hair_gaussians(self, n_hair_gaussians):
        util.set_uniform_1int(self.program, n_hair_gaussians, "n_hair_gaussians")

    def update_cutting_mode(self, cutting_mode):
        util.set_uniform_1int(self.program, int(cutting_mode), "cutting_mode")
    
    def update_coloring_mode(self, coloring_mode):
        util.set_uniform_1int(self.program, int(coloring_mode), "coloring_mode")

    def update_keep_sh(self, keep_sh):
        util.set_uniform_1int(self.program, int(keep_sh), "keep_sh")

    def update_selected_color(self, selected_color):
        util.set_uniform_v3(self.program, selected_color, "selected_color")

    def update_max_coloring_distance(self, max_coloring_distance):
        util.set_uniform_1f(self.program, max_coloring_distance, "max_coloring_distance")

    def update_invert_x_plane(self, invert_x_plane):
        util.set_uniform_1int(self.program, int(invert_x_plane), "invert_x_plane")

    def update_invert_y_plane(self, invert_y_plane):
        util.set_uniform_1int(self.program, int(invert_y_plane), "invert_y_plane")

    def update_invert_z_plane(self, invert_z_plane):
        util.set_uniform_1int(self.program, int(invert_z_plane), "invert_z_plane")

    def update_selected_head_avatar_index(self, selected_head_avatar_index):
        util.set_uniform_1int(self.program, selected_head_avatar_index, "selected_head_avatar_index")

    def update_max_cutting_distance(self, max_cutting_distance):
        util.set_uniform_1f(self.program, max_cutting_distance, "max_cutting_distance")

    def update_x_plane(self, x_plane):
        util.set_uniform_1f(self.program, x_plane, "x_plane")

    def update_y_plane(self, y_plane):
        util.set_uniform_1f(self.program, y_plane, "y_plane")

    def update_z_plane(self, z_plane):
        util.set_uniform_1f(self.program, z_plane, "z_plane")

    def update_ray_direction(self, camera: util.Camera, mouse_pos_2d):
        mouse_pos_3d = util.glhUnProjectf(mouse_pos_2d.x, mouse_pos_2d.y, 1, camera.get_view_matrix(), camera.get_project_matrix(), gl.glGetIntegerv(gl.GL_VIEWPORT))
        ray_direction = mouse_pos_3d-camera.position
        ray_direction = ray_direction/np.linalg.norm(ray_direction)
        util.set_uniform_v3(self.program, ray_direction, "ray_direction")

    def draw(self):
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)
        num_gau = len(self.gaussians)
        gl.glDrawArraysInstanced(gl.GL_LINES, 0, len(self.lines.reshape(-1)), num_gau)
