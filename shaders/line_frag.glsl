#version 430 core

in vec3 color;
in float alpha;

out vec4 FragColor;

void main()
{
    FragColor = vec4(color, alpha);
    return;
 }
