"""Replace one named part of an authored COLLADA physics document."""

from pathlib import Path
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET

import numpy as np

from lodestar_section_source import by_id, node_path, reference, scene_transform


COLLADA_NAMESPACE = "http://www.collada.org/2005/11/COLLADASchema"


def _position_ids(mesh):
    vertices = mesh.findall("vertices")
    if len(vertices) != 1 or not vertices[0].get("id"):
        raise ValueError("The physics part needs one original vertices ID")
    positions = vertices[0].findall("input[@semantic='POSITION']")
    if len(positions) != 1:
        raise ValueError("The physics part needs one original position stream")
    source_id = reference(positions[0].get("source", ""))
    source = by_id(mesh, "source", source_id)
    accessor = source.find("./technique_common/accessor")
    if accessor is None:
        raise ValueError("The physics part position stream needs an accessor")
    array_id = reference(accessor.get("source", ""))
    by_id(source, "float_array", array_id)
    return vertices[0].get("id"), source_id, array_id


def _replacement_mesh(mesh, vertices, indices):
    vertices_id, source_id, array_id = _position_ids(mesh)
    result = ET.Element("mesh")
    source = ET.SubElement(result, "source", id=source_id)
    array = ET.SubElement(source, "float_array", id=array_id, count=str(vertices.size))
    array.text = " ".join(format(value, ".9g") for value in vertices.ravel())
    common = ET.SubElement(source, "technique_common")
    accessor = ET.SubElement(common, "accessor", source=f"#{array_id}", count=str(len(vertices)), stride="3")
    for name in "XYZ":
        ET.SubElement(accessor, "param", name=name, type="float")
    vertex_input = ET.SubElement(result, "vertices", id=vertices_id)
    ET.SubElement(vertex_input, "input", semantic="POSITION", source=f"#{source_id}")
    triangles = ET.SubElement(result, "triangles", count=str(len(indices)))
    ET.SubElement(triangles, "input", semantic="VERTEX", source=f"#{vertices_id}", offset="0")
    ET.SubElement(triangles, "p").text = " ".join(map(str, indices.ravel()))
    return result


def _local_mesh(scene_triangles, translation, transform, reverse):
    scene_triangles = np.asarray(scene_triangles, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if (scene_triangles.ndim != 3 or scene_triangles.shape[1:] != (3, 3) or not len(scene_triangles) or
            translation.shape != (3,) or not np.isfinite(scene_triangles).all() or
            not np.isfinite(translation).all()):
        raise ValueError("Expected finite replacement triangles and one scene-space translation")
    if reverse:
        scene_triangles = scene_triangles[:, [0, 2, 1]]
    offsets = scene_triangles.reshape(-1, 3) + translation - transform[:3, 3]
    positions = np.linalg.solve(transform[:3, :3], offsets.T).T
    if not np.isfinite(positions).all() or np.max(np.abs(positions)) > np.finfo(np.float32).max:
        raise ValueError("Replacement coordinates exceed the COLLADA position range")
    local = positions.astype(np.float32)
    triangles = local.astype(np.float64).reshape(-1, 3, 3)
    edges = triangles[:, 1:] - triangles[:, :1]
    if np.any(np.all(np.cross(edges[:, 0], edges[:, 1]) == 0, axis=1)):
        raise ValueError("Replacement contains a zero-area triangle in the local coordinate frame")
    vertices, corners = np.unique(local, axis=0, return_inverse=True)
    return vertices, corners.reshape(-1, 3)


def replace_physics_part(
    source: Path,
    output: Path,
    node_name: str,
    aligned_scene_triangles: np.ndarray,
    translation: tuple[float, float, float],
):
    """Write a copy with one leaf's geometry rebuilt in its authored coordinate frame.

    The supplied positions use Y-up metre scene coordinates. Translation places them
    into the authored physics scene before the selected node's transform is inverted.
    """
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or (output.exists() and output.samefile(source)):
        raise ValueError("The replacement output would overwrite the authored physics")

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
    tree = ET.parse(source, parser=parser)
    root = tree.getroot()
    namespace = f"{{{COLLADA_NAMESPACE}}}"
    if root.tag != namespace + "COLLADA" or root.get("version") != "1.4.1":
        raise ValueError("The authored physics document must use COLLADA 1.4.1")

    original_tags = {}
    for element in root.iter():
        if isinstance(element.tag, str) and element.tag.startswith(namespace):
            original_tags[element] = element.tag
            element.tag = element.tag[len(namespace):]

    active_scenes = root.findall("./scene/instance_visual_scene")
    if len(active_scenes) != 1:
        raise ValueError("The authored physics document needs one active scene")
    scene = by_id(root, "./library_visual_scenes/visual_scene", reference(active_scenes[0].get("url", "")))
    nodes = node_path(scene, node_name)
    leaf = nodes[-1]
    instances = leaf.findall("instance_geometry")
    if len(instances) != 1 or any(leaf.find(tag) is not None for tag in ("node", "instance_node", "instance_controller")):
        raise ValueError("The physics part must be a leaf node with one geometry instance")
    node_ids = {f"#{node.get('id')}" for node in nodes if node.get("id")}
    if any(item.get("url") in node_ids for item in root.iter("instance_node")):
        raise ValueError("The physics part is in a shared COLLADA node")
    instance = instances[0]
    geometry_url = instance.get("url", "")
    geometry = by_id(root, "./library_geometries/geometry", reference(geometry_url))
    if any(element is not instance and geometry_url in (element.get("url"), element.get("source"))
           for element in root.iter()):
        raise ValueError("The physics part uses shared geometry")
    meshes = geometry.findall("mesh")
    if len(meshes) != 1:
        raise ValueError("The physics part needs one original mesh")
    mesh = meshes[0]
    transform, reverse = scene_transform(root, nodes)
    vertices, indices = _local_mesh(aligned_scene_triangles, translation, transform, reverse)
    replacement = _replacement_mesh(mesh, vertices, indices)
    mesh_members = set(mesh.iter())
    removed_ids = {f"#{item.get('id')}" for item in mesh_members if item.get("id")}
    removed_ids -= {f"#{item.get('id')}" for item in replacement.iter() if item.get("id")}
    if any(item not in mesh_members and removed_ids.intersection(item.attrib.values()) for item in root.iter()):
        raise ValueError("The physics part has externally referenced mesh data")

    location = list(geometry).index(mesh)
    geometry.remove(mesh)
    geometry.insert(location, replacement)
    for binding in instance.findall("bind_material"):
        instance.remove(binding)
    for element in replacement.iter():
        element.tag = namespace + element.tag
    for element, tag in original_tags.items():
        element.tag = tag
    ET.register_namespace("", COLLADA_NAMESPACE)

    temporary = None
    try:
        with NamedTemporaryFile(dir=output.parent, suffix=".dae", delete=False) as stream:
            temporary = Path(stream.name)
            tree.write(stream, encoding="utf-8", xml_declaration=True)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
