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
#define SCALE_IDX 7
#define OPACITY_IDX 10
#define SH_IDX 11

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
uniform int render_mod;

uniform int start_index;
uniform int n_hair_gaussians;
uniform int cutting_mode;
uniform float max_cutting_distance;
uniform int selected_head_avatar_index;
uniform vec3 ray_direction;

out vec3 color;
out float alpha;

mat3 computeSR(vec3 scale, vec4 q)  // should be correct
{
    mat3 S = mat3(0.f);
    S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

    mat3 R = mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

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

void main()
{
	int boxid = gi[gl_InstanceID];
	int total_dim = 3 + 4 + 3 + 1 + sh_dim;
	int start = boxid * total_dim;
	vec4 g_pos = vec4(get_vec3(start + POS_IDX), 1.f);
    vec4 g_pos_view = view_matrix * g_pos;
    vec4 g_pos_screen = projection_matrix * g_pos_view;
	g_pos_screen.xyz = g_pos_screen.xyz / g_pos_screen.w;
    g_pos_screen.w = 1.f;
	float g_opacity = g_data[start + OPACITY_IDX];
	// early culling
	if (any(greaterThan(abs(g_pos_screen.xyz), vec3(1.3))) || g_opacity == 0.f)
	{
		gl_Position = vec4(-100, -100, -100, 1);
		return;
	}
	vec4 g_rot = get_vec4(start + ROT_IDX);
	vec3 g_scale = get_vec3(start + SCALE_IDX);

	mat3 M = computeSR(g_scale * scale_modifier, g_rot);
	vec4 second_point = vec4(lines*M + g_pos.xyz, 1.f);
	vec4 second_point_view = view_matrix * second_point;
	vec4 second_point_screen = projection_matrix * second_point_view;
	gl_Position = second_point_screen;
	color = abs(lines);
	alpha = 1.;

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
		alpha = distance < max_cutting_distance ? 0 : 1;
	}

	if (render_mod == -1)
	{
		float depth = -g_pos_view.z;
		depth = depth < 0.05 ? 1 : depth;
		alpha = 1 / depth;
	}
}
