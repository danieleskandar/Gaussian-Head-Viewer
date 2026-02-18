#version 430 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 lines;
// layout(location = 1) in vec3 g_pos;
// layout(location = 2) in vec4 g_rot;
// layout(location = 3) in vec3 g_scale;
// layout(location = 4) in vec3 g_dc_color;
// layout(location = 5) in float g_opacity;

#define POS_IDX 0
#define ROT_IDX 3
#define SCALE_IDX 12      // 3 (pos) + 9 (rot)
#define OPACITY_IDX 15    // 12 + 3 (scale)
#define SH_IDX 16         // 15 + 1 (opacity)

layout (std430, binding=0) buffer gaussian_data {
	float g_data[];
	// compact version of following data
	// vec3 g_pos[];
	// vec4 g_rot[];
	// vec3 g_scale[];
	// float g_opacity[];
	// vec3 g_sh[];
};
layout (std430, binding=1) buffer gaussian_order {
	int gi[];
};

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec3 cam_pos;
uniform int sh_dim;
uniform float scale_modifier;
uniform int render_mod;  // > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 gaussian

uniform int start_index;
uniform int n_gaussians;
uniform int n_hair_gaussians;
uniform int cutting_mode;
uniform float max_cutting_distance;
uniform int invert_x_plane;
uniform float x_plane;
uniform int invert_y_plane;
uniform float y_plane;
uniform int invert_z_plane;
uniform float z_plane;
uniform int selected_head_avatar_index;
uniform vec3 ray_direction;

out vec3 color;
out float alpha;

mat3 computeSR(vec3 scale, mat3 R)
{
    mat3 S = mat3(0.f);
    S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;
	
    mat3 M = S * R;
    return M;
}

vec3 get_vec3(int offset)
{
	return vec3(g_data[offset], g_data[offset + 1], g_data[offset + 2]);
}
vec4 get_vec4(int offset)
{
	return vec4(g_data[offset], g_data[offset + 1], g_data[offset + 2], g_data[offset + 3]);
}

mat3 get_rot_quat(int offset) {
    vec4 q = get_vec4(offset);
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;
    return mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z), 2.f * (x * z - r * y),
        2.f * (x * y - r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x),
        2.f * (x * z + r * y), 2.f * (y * z - r * x), 1.f - 2.f * (x * x + y * y)
    );
}

mat3 get_rot_mat(int offset) {
    vec3 col0 = get_vec3(offset);
    vec3 col1 = get_vec3(offset + 3);
    vec3 col2 = get_vec3(offset + 6);
    return mat3(col0, col1, col2);
}

bool is_quat(int offset) {
    // Check the last 5 floats (indices 4, 5, 6, 7, 8)
    // We can check a few; checking all 5 is safest.
    float v4 = g_data[offset + 4];
    float v5 = g_data[offset + 5];
    float v6 = g_data[offset + 6];
    float v7 = g_data[offset + 7];
    float v8 = g_data[offset + 8];

    // Using exact 0.0 check is safe here because we explicitly write 0.0 in Python
    return (v4 < 0.0 && v5 < 0.0 && v6 < 0.0 && v7 < 0.0 && v8 < 0.0);
}


void main()
{
	int boxid = gi[gl_InstanceID];
	int total_dim = 3 + 9 + 3 + 1 + sh_dim;
	int start = boxid * total_dim;
	vec4 g_pos = vec4(get_vec3(start + POS_IDX), 1.f);
    vec4 g_pos_view = view_matrix * g_pos;
    vec4 g_pos_screen = projection_matrix * g_pos_view;
	g_pos_screen.xyz = g_pos_screen.xyz / g_pos_screen.w;
    g_pos_screen.w = 1.f;
	// early culling
	if (any(greaterThan(abs(g_pos_screen.xyz), vec3(1.3))))
	{
		gl_Position = vec4(-100, -100, -100, 1);
		return;
	}

	mat3 R;
    int rot_offset = start + ROT_IDX;

    if (is_quat(rot_offset)) {
        R = get_rot_quat(rot_offset);
    } else {
        R = get_rot_mat(rot_offset);
    }
	
	vec3 g_scale = get_vec3(start + SCALE_IDX);
	float g_opacity = g_data[start + OPACITY_IDX];

    mat3 M = computeSR(g_scale * scale_modifier, R);
	vec4 second_point = vec4(lines*M + g_pos.xyz, 1.f);
	vec4 second_point_view = view_matrix * second_point;
	vec4 second_point_screen = projection_matrix * second_point_view;
	gl_Position = second_point_screen;
	color = abs(lines);
	if (g_opacity == 0.0) {
		alpha = 0.0;
	} else {
		alpha = 1.0;
	}

	if (render_mod == -5)
	{
		float projection = dot(ray_direction, g_pos.xyz-cam_pos);
		vec3 closest_point = cam_pos + projection * ray_direction;
		float distance = length(g_pos.xyz - closest_point);
		alpha = distance < 0.2 ? 1 : 0.2;
	}

	if (cutting_mode == 1 && selected_head_avatar_index > -1 && boxid >= start_index && boxid < start_index + n_hair_gaussians) {	
		float projection = dot(ray_direction, g_pos.xyz-cam_pos);
		vec3 closest_point = cam_pos + projection * ray_direction;
		float distance = length(g_pos.xyz - closest_point);
		alpha = distance < max_cutting_distance ? 0 : alpha;
	}

	if (selected_head_avatar_index > -1 && boxid >= start_index && boxid < (start_index + n_gaussians)) {
		if (invert_x_plane == 0 && g_pos.x >= x_plane) alpha = 0;
		if (invert_x_plane == 1 && g_pos.x <= x_plane) alpha = 0;
		if (invert_y_plane == 0 && g_pos.y >= y_plane) alpha = 0;
		if (invert_y_plane == 1 && g_pos.y <= y_plane) alpha = 0;
		if (invert_z_plane == 0 && g_pos.z >= z_plane) alpha = 0;
		if (invert_z_plane == 1 && g_pos.z <= z_plane) alpha = 0;
	}

	if (render_mod == -1)
	{
		float depth = -g_pos_view.z;
		depth = depth < 0.05 ? 1 : depth;
		alpha = 1 / depth;
	}
}
