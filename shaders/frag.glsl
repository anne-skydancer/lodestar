#version 450

layout(location = 0) in vec3 normal;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 direction = normalize(vec3(0.4, 0.8, 1.0));
    float diffuse = max(dot(normalize(normal), direction), 0.0);
    vec3 base = vec3(0.32, 0.53, 0.80);
    outColor = vec4(base * (0.25 + 0.75 * diffuse), 1.0);
}
