"""Measure matched offscreen GPU draws for Lodestar's embedded-texture GLBs."""

import argparse
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from lodestar_section_compare import AMBIENT, BACKGROUND, LIGHT, VIEWS


COLOR_FORMAT = "rgba8unorm"
DEPTH_FORMAT = "depth32float"
SHADER = """
struct Camera {
    transform: mat4x4<f32>,
    light_and_ambient: vec4f,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var diffuse_map: texture_2d<f32>;
@group(1) @binding(1) var diffuse_sampler: sampler;

struct Interpolated {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) uv: vec2f,
};

@vertex
fn vs_main(@location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f) -> Interpolated {
    var output: Interpolated;
    output.position = camera.transform * vec4f(position, 1.0);
    output.normal = normal;
    output.uv = uv;
    return output;
}

@fragment
fn fs_main(input: Interpolated) -> @location(0) vec4f {
    let normal = input.normal * inverseSqrt(max(dot(input.normal, input.normal), 1e-20));
    let light = camera.light_and_ambient;
    let illumination = light.w + (1.0 - light.w) * clamp(dot(normal, light.xyz), 0.0, 1.0);
    let linear = illumination * textureSample(diffuse_map, diffuse_sampler, input.uv).rgb;
    let rgb = select(1.055 * pow(linear, vec3f(1.0 / 2.4)) - 0.055, 12.92 * linear, linear < vec3f(0.0031308));
    return vec4f(rgb, 1.0);
}
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class Model:
    positions: np.ndarray
    bundle: object
    stats: dict


def load_model(path, device, pipeline, camera, wgpu):
    data = path.read_bytes()
    magic, version, length, size, kind = struct.unpack_from("<4sIIII", data)
    require((magic, version, length, kind) == (b"glTF", 2, len(data), 0x4E4F534A), f"Invalid GLB {path}")
    document = json.loads(data[20:20 + size])
    binary_size, kind = struct.unpack_from("<II", data, 20 + size)
    binary = memoryview(data)[28 + size:]
    require(kind == 0x004E4942 and binary_size == len(binary), f"Expected an embedded buffer in {path}")
    require(document.get("nodes") == [{"mesh": 0}] and len(document["meshes"]) == 1,
            f"Expected a single untransformed Lodestar mesh in {path}")

    def buffer_view(identifier):
        view = document["bufferViews"][identifier]
        offset = view.get("byteOffset", 0)
        require(view["buffer"] == 0 and "byteStride" not in view and
                0 <= offset < offset + view["byteLength"] <= len(binary), f"Unsupported buffer view in {path}")
        return binary[offset:offset + view["byteLength"]]

    def accessor(identifier, expected_type, dtype):
        entry = document["accessors"][identifier]
        component, width = {"SCALAR": (None, 1), "VEC2": (5126, 2), "VEC3": (5126, 3)}[expected_type]
        require(entry["type"] == expected_type and (component is None or entry["componentType"] == component) and
                "sparse" not in entry and not entry.get("normalized", False), f"Unsupported accessor in {path}")
        return np.frombuffer(buffer_view(entry["bufferView"]), dtype=dtype, count=entry["count"] * width,
                             offset=entry.get("byteOffset", 0)).reshape(-1, width)

    positions, images, bind_groups = [], {}, {}
    draw_calls = 0
    image_rgba8_bytes = 0
    geometry_bytes, triangle_count = 0, 0
    commands = device.create_render_bundle_encoder(color_formats=[COLOR_FORMAT], depth_stencil_format=DEPTH_FORMAT)
    commands.set_pipeline(pipeline)
    commands.set_bind_group(0, camera)
    diffuse_layout = pipeline.get_bind_group_layout(1)
    for primitive in document["meshes"][0]["primitives"]:
        require(primitive.get("mode", 4) == 4, f"Expected indexed triangles in {path}")
        attrs = primitive["attributes"]
        xyz = accessor(attrs["POSITION"], "VEC3", "<f4")
        normals = accessor(attrs["NORMAL"], "VEC3", "<f4")
        uvs = accessor(attrs["TEXCOORD_0"], "VEC2", "<f4")
        index_type = document["accessors"][primitive["indices"]]["componentType"]
        index_dtype = {5123: "<u2", 5125: "<u4"}.get(index_type)
        require(index_dtype is not None, f"Expected unsigned 16- or 32-bit indices in {path}")
        indices = accessor(primitive["indices"], "SCALAR", index_dtype).ravel()
        require(len(indices) > 0 and len(indices) % 3 == 0 and indices.max() < len(xyz),
                f"Invalid triangle indices in {path}")
        material = document["materials"][primitive["material"]]
        require(not material.get("extensions"), f"Expected a diffuse-only material in {path}")
        binding = material["pbrMetallicRoughness"]["baseColorTexture"]
        require(binding.get("texCoord", 0) == 0, f"Expected UV0 in {path}")
        texture_id = binding["index"]
        if texture_id not in bind_groups:
            texture = document["textures"][texture_id]
            image_id = texture["source"]
            if image_id not in images:
                image = document["images"][image_id]
                with Image.open(BytesIO(buffer_view(image["bufferView"]))) as decoded:
                    pixels = decoded.convert("RGBA")
                    image_rgba8_bytes += pixels.width * pixels.height * 4
                    dimensions = (*pixels.size, 1)
                    # The sRGB format decodes the source to linear light before filtering.
                    image_texture = device.create_texture(size=dimensions, format="rgba8unorm-srgb",
                                                          usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
                    device.queue.write_texture({"texture": image_texture}, pixels.tobytes(),
                                               {"bytes_per_row": 4 * pixels.width, "rows_per_image": pixels.height}, dimensions)
                images[image_id] = image_texture.create_view()
            sampler = document["samplers"][texture["sampler"]]
            require(sampler["magFilter"] == sampler["minFilter"] == 9729 and
                    sampler["wrapS"] in (10497, 33071) and sampler["wrapT"] in (10497, 33071),
                    f"Expected bilinear repeat or clamp sampling in {path}")
            wrap = {10497: "repeat", 33071: "clamp-to-edge"}
            gpu_sampler = device.create_sampler(mag_filter="linear", min_filter="linear",
                                                address_mode_u=wrap[sampler["wrapS"]], address_mode_v=wrap[sampler["wrapT"]])
            bind_groups[texture_id] = device.create_bind_group(layout=diffuse_layout, entries=[
                {"binding": 0, "resource": images[image_id]}, {"binding": 1, "resource": gpu_sampler},
            ])

        vertices = np.column_stack((xyz, normals, uvs))
        vertex_buffer = device.create_buffer_with_data(data=vertices, usage=wgpu.BufferUsage.VERTEX)
        index_buffer = device.create_buffer_with_data(data=indices, usage=wgpu.BufferUsage.INDEX)
        commands.set_bind_group(1, bind_groups[texture_id])
        commands.set_vertex_buffer(0, vertex_buffer)
        commands.set_index_buffer(index_buffer, "uint16" if indices.itemsize == 2 else "uint32")
        commands.draw_indexed(len(indices))
        draw_calls += 1
        positions.append(xyz)
        triangle_count += len(indices) // 3
        geometry_bytes += vertices.nbytes + indices.nbytes
    require(draw_calls > 0, f"No drawable triangles in {path}")
    return Model(np.concatenate(positions), commands.finish(),
                 {"path": str(path), "triangles": triangle_count, "draw_calls": draw_calls,
                  "embedded_images": len(images), "image_rgba8_bytes": image_rgba8_bytes,
                  "geometry_bytes": geometry_bytes})


def camera_matrix(positions, direction):
    toward = np.asarray(direction, dtype=np.float64)
    toward /= np.linalg.norm(toward)
    reference_up = [0, 1, 0] if abs(toward[1]) < 0.99 else [0, 0, -1]
    right = np.cross(reference_up, toward)
    right /= np.linalg.norm(right)
    up = np.cross(toward, right)
    basis = np.stack((right, up, toward))
    projected = positions @ basis.T
    minimum, maximum = projected.min(axis=0), projected.max(axis=0)
    span = maximum - minimum
    require(np.isfinite(span).all() and max(span[:2]) > 0, "Invalid shared frame bounds")
    scale = np.array([1.8, 1.8, 0.9]) / np.array([max(span[:2]), max(span[:2]), span[2] or 1.0])
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = basis * scale[:, None]
    matrix[:3, 3] = -(minimum + maximum) * scale / 2
    matrix[2, 3] += 0.5
    return matrix.T.tobytes() + np.array([*LIGHT, AMBIENT], dtype=np.float32).tobytes()


def draw(model, encoder, color, depth):
    red, green, blue = BACKGROUND
    render = encoder.begin_render_pass(color_attachments=[{
        "view": color, "clear_value": (red / 255, green / 255, blue / 255, 0.0),
        "load_op": "clear", "store_op": "store",
    }], depth_stencil_attachment={
        "view": depth, "depth_clear_value": 0.0, "depth_load_op": "clear", "depth_store_op": "discard",
    })
    render.execute_bundles([model.bundle])
    render.end()


def stats(samples):
    return {"median_us": float(np.median(samples)), "p25_us": float(np.percentile(samples, 25)),
            "p75_us": float(np.percentile(samples, 75))}


def benchmark(args, wgpu):
    from wgpu.backends.wgpu_native.extras import set_instance_extras, write_timestamp

    set_instance_extras(backends=(args.backend,), flags=("Default", "AutomaticTimestampNormalization"))
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync(required_features=["timestamp-query", "timestamp-query-inside-encoders"])
    shader = device.create_shader_module(code=SHADER)
    pipeline = device.create_render_pipeline(layout="auto", vertex={
        "module": shader, "entry_point": "vs_main", "buffers": [{
            "array_stride": 32, "step_mode": "vertex", "attributes": [
                {"format": "float32x3", "offset": 0, "shader_location": 0},
                {"format": "float32x3", "offset": 12, "shader_location": 1},
                {"format": "float32x2", "offset": 24, "shader_location": 2},
            ],
        }],
    }, primitive={"topology": "triangle-list", "cull_mode": "none"}, depth_stencil={
        "format": DEPTH_FORMAT, "depth_write_enabled": True, "depth_compare": "greater",
    }, fragment={"module": shader, "entry_point": "fs_main", "targets": [{"format": COLOR_FORMAT}]})
    camera_buffer = device.create_buffer(size=80, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    camera = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[{
        "binding": 0, "resource": {"buffer": camera_buffer, "offset": 0, "size": 80},
    }])
    source = load_model(args.source, device, pipeline, camera, wgpu)
    candidate = load_model(args.candidate, device, pipeline, camera, wgpu)
    models = (source, candidate)
    positions = np.concatenate((source.positions, candidate.positions)).astype(np.float64)
    dimensions = (args.resolution, args.resolution, 1)
    color_texture = device.create_texture(size=dimensions, format=COLOR_FORMAT,
                                          usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    color = color_texture.create_view()
    depth = device.create_texture(size=dimensions, format=DEPTH_FORMAT,
                                  usage=wgpu.TextureUsage.RENDER_ATTACHMENT).create_view()
    query_count = args.repeats * len(models) * 2
    queries = device.create_query_set(type="timestamp", count=query_count)
    resolved = device.create_buffer(size=query_count * 8, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)

    results = {}
    for view, direction in VIEWS:
        device.queue.write_buffer(camera_buffer, 0, camera_matrix(positions, direction))
        coverage = []
        for model in models:
            encoder = device.create_command_encoder()
            draw(model, encoder, color, depth)
            device.queue.submit([encoder.finish()])
            rgba = device.queue.read_texture({"texture": color_texture}, {"bytes_per_row": args.resolution * 4}, dimensions)
            rendered = np.frombuffer(rgba, dtype=np.uint8).reshape(-1, 4)
            coverage.append(int(np.count_nonzero(rendered[:, 3])))
        warmup = device.create_command_encoder()
        for _ in range(args.warmups):
            for model in models:
                draw(model, warmup, color, depth)
        device.queue.submit([warmup.finish()])

        # Record the entire sequence before submission so the timestamp interval
        # does not depend on Python keeping up with the GPU.
        encoder = device.create_command_encoder()
        for repeat in range(args.repeats):
            order = (0, 1) if repeat % 2 == 0 else (1, 0)
            for index in order:
                slot = (repeat * len(models) + index) * 2
                write_timestamp(encoder, queries, slot)
                for _ in range(args.batch):
                    draw(models[index], encoder, color, depth)
                write_timestamp(encoder, queries, slot + 1)
        encoder.resolve_query_set(queries, 0, query_count, resolved, 0)
        device.queue.submit([encoder.finish()])
        timestamps = np.frombuffer(device.queue.read_buffer(resolved), dtype="<u8").reshape(args.repeats, len(models), 2)
        require(np.all(timestamps[..., 1] > timestamps[..., 0]), "Invalid GPU timer result")
        elapsed = (timestamps[..., 1] - timestamps[..., 0]).T
        samples = (elapsed / (1000.0 * args.batch)).tolist()
        results[view] = {
            label: {**stats(samples[index]), "covered_pixels": coverage[index], "batch_samples_us": samples[index]}
            for index, label in enumerate(("source", "candidate"))
        }

    require(all(any(view[label]["covered_pixels"] for view in results.values()) for label in ("source", "candidate")),
            "One of the meshes rendered no pixels in any view")
    batch_means = {label: np.mean([result[label]["batch_samples_us"] for result in results.values()], axis=0)
                   for label in ("source", "candidate")}
    overall = {label: stats(samples) for label, samples in batch_means.items()}
    source_time, candidate_time = (overall[label]["median_us"] for label in ("source", "candidate"))
    overall["candidate_change_percent"] = (candidate_time / source_time - 1) * 100
    paired_change = (batch_means["candidate"] / batch_means["source"] - 1) * 100
    bootstrap_pair_ids = np.random.default_rng(12).integers(args.repeats, size=(4096, args.repeats))
    bootstrap_changes = np.median(paired_change[bootstrap_pair_ids], axis=1)
    overall["paired_change_percent"] = float(np.median(paired_change))
    overall["paired_change_95pct_ci_percent"] = np.percentile(bootstrap_changes, [2.5, 97.5]).tolist()
    return {"renderer": adapter.info["device"], "adapter": dict(adapter.info),
            "resolution": [args.resolution, args.resolution],
            "method": "GPU timestamps for prerecorded offscreen framebuffer clears and indexed textured draws",
            "warmups_per_mesh_and_view": args.warmups, "batches_per_mesh_and_view": args.repeats,
            "frames_per_batch": args.batch, "source": source.stats, "candidate": candidate.stats,
            "overall": overall, "views": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--backend", choices=("Vulkan", "DX12", "Metal", "GL"), default="Vulkan")
    args = parser.parse_args()
    if min(args.resolution, args.warmups, args.repeats, args.batch) <= 0:
        parser.error("Resolution, warmup count, repeat count and batch size must be positive")
    try:
        import wgpu
    except ModuleNotFoundError:
        parser.exit(1, "The GPU benchmark needs wgpu 0.33 and its dependencies.\n")
    try:
        report = benchmark(args, wgpu)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, RuntimeError, struct.error, wgpu.GPUError) as error:
        parser.exit(1, f"Textured GLB benchmark: {error}\n")
    print(json.dumps({"renderer": report["renderer"], "overall": report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
