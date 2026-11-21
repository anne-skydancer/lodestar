"""Fetch the public reference files used by the Lodestar benchmark."""

import argparse
import hashlib
import io
from pathlib import Path
import struct
from urllib.request import Request, urlopen
from zipfile import ZipFile


# Second Life Wiki Mesh Import test attachments are used for local evaluation.
ARCHIVES = (
    (
        "https://github.com/user-attachments/files/18443046/duck.zip",
        "053899a41b1c6bc7c71e0f18ece3ba3dee578c8aff1f3c940b4bc7fe772b9b5d",
        ("duck.dae",),
    ),
    (
        "https://github.com/user-attachments/files/18457327/TestHall_models_new_2.zip",
        "ea364444d21296c20656b80a92e4324204c4765606f899490da5b5d64e91dbb0",
        ("TestHall_MESH.dae", "TestHall_LOD2.dae", "TestHall_PHYS.dae"),
    ),
    (
        "https://github.com/user-attachments/files/18457007/simplebot.zip",
        "dab9846e75994906e5505661eb73306b55b38db6f1ed74946084075a1064e517",
        ("simplebot.dae",),
    ),
    (
        "https://github.com/user-attachments/files/18457307/wholey.sphere.zip",
        "96ab938a154c5db3e06afcb7c8b38474d13a3c9d86395e8a5bad05625165513a",
        ("wholey sphere.dae",),
    ),
)

# Khronos glTF Sample Assets, Barramundi Fish by Microsoft, CC0 1.0.
FISH_URL = (
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/"
    "0b4f3a5862f3037c7092206e75c6f2a31a504ca5/Models/BarramundiFish/glTF-Binary/BarramundiFish.glb"
)
FISH_SHA256 = "ecc3bafb6b00f2c8b810863c388e3768a7b7ea0d0335e8cb8c574c266e571f4a"

# Wikimedia Commons Sphere (Icosphere), The People's Internet, CC0 1.0.
SPHERE_URL = "https://upload.wikimedia.org/wikipedia/commons/1/11/Sphere_%28Icosphere%29.stl"
SPHERE_SHA256 = "82c854ba54794ed154244bd67667e8e8011c7145f92d64fd8a1c969919679596"


def download(url, expected_sha256):
    request = Request(url, headers={"User-Agent": "Lodestar mesh benchmark/1.0"})
    with urlopen(request, timeout=45) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"Unexpected SHA-256 for {url}")
    return data


def copy_stl_geometry_as_collada(data):
    """Copy binary STL positions and winding into the supported COLLADA format."""
    header_size = 84
    triangle = struct.Struct("<12fH")
    if len(data) < header_size:
        raise ValueError("Expected a binary STL header")
    (triangle_count,) = struct.unpack_from("<I", data, 80)
    if len(data) != header_size + triangle_count * triangle.size:
        raise ValueError("Unexpected binary STL size")

    vertices = {}
    indices = []
    for record in triangle.iter_unpack(memoryview(data)[header_size:]):
        for start in (3, 6, 9):
            point = record[start : start + 3]
            indices.append(vertices.setdefault(point, len(vertices)))

    # Nine significant digits preserve every float32 coordinate on round-trip.
    positions = " ".join(format(component, ".9g") for point in vertices for component in point)
    index_text = " ".join(map(str, indices))
    return f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit meter="1" name="meter"/><up_axis>Z_UP</up_axis></asset>
  <library_geometries><geometry id="icosphere"><mesh>
    <source id="positions">
      <float_array id="positions-array" count="{3 * len(vertices)}">{positions}</float_array>
      <technique_common><accessor source="#positions-array" count="{len(vertices)}" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
      </accessor></technique_common>
    </source>
    <vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
    <triangles count="{triangle_count}"><input semantic="VERTEX" source="#vertices" offset="0"/><p>{index_text}</p></triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="scene"><node id="shape">
    <instance_geometry url="#icosphere"/>
  </node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
""".encode("utf-8")


def write_asset(path, data):
    try:
        with path.open("xb") as output:
            output.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError(f"Existing file has different contents: {path}") from None
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory for the benchmark inputs")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for url, sha256, filenames in ARCHIVES:
        with ZipFile(io.BytesIO(download(url, sha256))) as archive:
            for filename in filenames:
                write_asset(args.directory / filename, archive.read(filename))
    write_asset(args.directory / "BarramundiFish.glb", download(FISH_URL, FISH_SHA256))
    sphere = copy_stl_geometry_as_collada(download(SPHERE_URL, SPHERE_SHA256))
    write_asset(args.directory / "Icosphere.dae", sphere)


if __name__ == "__main__":
    main()
