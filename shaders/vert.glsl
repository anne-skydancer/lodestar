#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 0) out vec3 normal;

layout(push_constant) uniform Transforms {
    mat4 mvp;
    mat4 model;
} transforms;

void main() {
    gl_Position = transforms.mvp * vec4(inPosition, 1.0);
    normal = mat3(transforms.model) * inNormal;
}
