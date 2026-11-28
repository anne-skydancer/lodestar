"""Check the C++ correspondenceBound against the Python correspondence_bound on shared-UV fixtures."""

import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from lodestar_surface_fit import correspondence_bound  # noqa: E402


def lattice(n):
    return lambda i, j: j * (n + 1) + i


def grid(n, jitter, seed):
    """A heightfield chart with fine triangles and two coarse retriangulations (same vertices)."""
    rng = np.random.default_rng(seed)
    uv = np.array([(i / n, j / n) for j in range(n + 1) for i in range(n + 1)])
    inner = (uv > 1e-9).all(axis=1) & (uv < 1 - 1e-9).all(axis=1)
    uv[inner] += rng.uniform(-jitter, jitter, size=(inner.sum(), 2)) / n
    height = 0.2 * np.sin(5 * uv[:, 0]) * np.cos(4 * uv[:, 1]) + 0.3 * uv[:, 0] * uv[:, 1]
    positions = np.column_stack((uv[:, 0] * 2.0, height, uv[:, 1] * 1.5))
    gx = 0.2 * 5 * np.cos(5 * uv[:, 0]) * np.cos(4 * uv[:, 1]) + 0.3 * uv[:, 1]
    gz = -0.2 * 4 * np.sin(5 * uv[:, 0]) * np.sin(4 * uv[:, 1]) + 0.3 * uv[:, 0]
    normals = np.column_stack((-gx, np.ones(len(uv)), -gz))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    ident = lattice(n)
    fine, same, flipped = [], [], []
    for j in range(n):
        for i in range(n):
            a, b, c, d = ident(i, j), ident(i + 1, j), ident(i + 1, j + 1), ident(i, j + 1)
            fine += [(a, b, c), (a, c, d)]
            same += [(a, b, c), (a, c, d)]
            flipped += [(a, b, d), (b, c, d)]
    return uv, positions, normals, np.array(fine), np.array(same), np.array(flipped)


def nested(n, factor):
    """A coarse triangulation on every `factor`-th lattice vertex of the same vertex array."""
    uv, positions, normals, fine, _, _ = grid(n, 0.0, 0)
    ident = lattice(n)
    coarse = []
    for j in range(0, n, factor):
        for i in range(0, n, factor):
            a, b, c, d = ident(i, j), ident(i + factor, j), ident(i + factor, j + factor), ident(i, j + factor)
            coarse += [(a, b, c), (a, c, d)]
    return uv, positions, normals, fine, np.array(coarse)


def write(path, uv, positions, normals, source, target):
    with open(path, "w") as f:
        f.write(f"V {len(uv)}\n")
        for t, p, n in zip(uv, positions, normals):
            f.write(" ".join(f"{x:.17g}" for x in (*t, *p, *n)) + "\n")
        for tag, faces in (("S", source), ("T", target)):
            f.write(f"{tag} {len(faces)}\n")
            for face in faces:
                f.write(" ".join(str(int(v)) for v in face) + "\n")


def run(exe, path):
    out = subprocess.run([exe, path], capture_output=True, text=True, check=True).stdout.split()
    return float(out[0]), float(out[1]), int(out[2]), int(out[3]), int(out[4])


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main():
    exe = sys.argv[1]
    fixtures = []
    uv, pos, nrm, fine, same, flipped = grid(8, 0.0, 1)
    fixtures.append(("aligned grid, flipped diagonals", uv, pos, nrm, flipped, fine))
    uv, pos, nrm, fine, same, flipped = grid(10, 0.28, 2)
    fixtures.append(("jittered grid, flipped diagonals", uv, pos, nrm, flipped, fine))
    uv, pos, nrm, fine, same, flipped = grid(12, 0.2, 3)
    fixtures.append(("jittered grid, same triangles", uv, pos, nrm, same, fine))
    uv, pos, nrm, fine, coarse = nested(8, 2)
    fixtures.append(("nested 8 -> 4", uv, pos, nrm, coarse, fine))
    uv, pos, nrm, fine, coarse = nested(12, 4)
    fixtures.append(("nested 12 -> 3", uv, pos, nrm, coarse, fine))
    uv, pos, nrm, fine, coarse = nested(8, 2)
    fixtures.append(("constant normals", uv, pos, np.tile([0.0, 1.0, 0.0], (len(uv), 1)), coarse, fine))
    uv, pos, nrm, fine, same, flipped = grid(9, 0.25, 4)
    rng = np.random.default_rng(5)
    wild = nrm + 0.4 * rng.normal(size=nrm.shape)
    wild /= np.linalg.norm(wild, axis=1)[:, None]
    fixtures.append(("strongly varying normals", uv, pos, wild, flipped, fine))
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, uv, pos, nrm, source, target in fixtures:
            path = os.path.join(tmp, "fixture.txt")
            write(path, uv, pos, nrm, source, target)
            expected = correspondence_bound(uv, pos, nrm, source, target)
            surface, angle, pairs, covered, limit = run(exe, path)
            ok = (covered == 1 and limit == 0 and pairs == expected["overlay_pairs"]
                  and close(surface, expected["surface_bound"]) and close(angle, expected["normal_angle_bound_degrees"]))
            print(f"{'ok  ' if ok else 'FAIL'} {name}: surface {surface:.12g} vs {expected['surface_bound']:.12g}, "
                  f"angle {angle:.12g} vs {expected['normal_angle_bound_degrees']:.12g}, pairs {pairs} vs {expected['overlay_pairs']}")
            failures += not ok
        # Coverage: the Python raises where the source leaves part of the domain uncovered.
        uv, pos, nrm, fine, same, flipped = grid(6, 0.0, 6)
        path = os.path.join(tmp, "partial.txt")
        write(path, uv, pos, nrm, fine, fine[: len(fine) // 2])
        try:
            correspondence_bound(uv, pos, nrm, fine, fine[: len(fine) // 2])
            raised = False
        except ValueError:
            raised = True
        covered_whole = run(exe, path)[3]
        write(path, uv, pos, nrm, fine[: len(fine) // 2], fine)
        covered_half = run(exe, path)[3]
        ok = raised and covered_whole == 0 and covered_half == 1
        print(f"{'ok  ' if ok else 'FAIL'} coverage: python raised={raised}, C++ covered flags {covered_whole}/{covered_half}")
        failures += not ok
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
