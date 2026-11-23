"""Replay explicit walking routes against a static, Y-up triangle collider."""

import math

import numpy as np

from check_testhall_physics import CAPSULE_HEIGHT, CAPSULE_RADIUS, floor_height, route_endpoints
from lodestar_collision_probes import sweep_clearance


def walk_route(triangles, start_feet, waypoints_xz, *, radius, height,
               max_step_up=0.22, max_step_down=0.22, skin=0.005, max_floor_slope=45.0):
    """Replay lift, forward travel, and landing between supported XZ waypoints.

    Waypoints are explicit landing locations, for example near the center of a
    tread. Capsule motion is swept continuously. Ground support is additionally
    sampled every 5 cm or less between waypoints. Positions are in meters.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    start = np.asarray(start_feet, dtype=np.float64)
    route = np.asarray(waypoints_xz, dtype=np.float64)
    if route.shape == (0,):
        route = route.reshape(0, 2)
    values = (radius, height, max_step_up, max_step_down, skin, max_floor_slope)
    if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or start.shape != (3,) or
            route.ndim != 2 or route.shape[1] != 2 or not all(map(math.isfinite, values)) or
            not all(np.isfinite(array).all() for array in (triangles, start, route)) or
            radius <= 0 or height < 2 * radius or min(max_step_up, max_step_down, skin) < 0 or
            not 0 <= max_floor_slope < 90):
        raise ValueError("Expected finite collider triangles, XYZ start, XZ waypoints and valid capsule limits")

    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    supporting = triangles[(lengths > 0) & (normals[:, 1] >= lengths * math.cos(math.radians(max_floor_slope)))]
    report = {"passed": False, "radius_m": radius, "height_m": height,
              "max_step_up_m": max_step_up, "max_step_down_m": max_step_down, "skin_m": skin,
              "feet_trace_m": [], "steps": []}
    epsilon = 1e-7

    def support(xz, low, high):
        return floor_height(supporting, *xz, low - epsilon, high + epsilon)

    def clearance(first, second):
        return sweep_clearance(triangles, first, second, radius, height)

    floor = support(start[[0, 2]], start[1] - max_step_down, start[1] + skin)
    if floor is None:
        report["failure"] = "start_has_no_support"
        return report
    feet = np.array([start[0], floor + skin, start[2]])
    if clearance(start, feet) < -epsilon:
        report["failure"] = "start_is_obstructed"
        return report
    report["feet_trace_m"].append(feet.tolist())

    for index, xz in enumerate(route):
        y = feet[1] - skin
        next_floor = support(xz, y - max_step_down, y + max_step_up)
        if next_floor is None:
            report.update(failure="landing_has_no_support_within_step_limits", failed_waypoint=index)
            return report
        landing = np.array([xz[0], next_floor + skin, xz[1]])
        raised_y = max(feet[1], landing[1])
        raised = np.array([feet[0], raised_y, feet[2]])
        above_landing = np.array([landing[0], raised_y, landing[2]])
        report["steps"].append({"rise_m": float(next_floor - y), "support_y_m": next_floor})
        step = report["steps"][-1]
        for phase, first, second in (("lift", feet, raised), ("forward", raised, above_landing),
                                     ("landing", above_landing, landing)):
            gap = clearance(first, second)
            step[phase + "_clearance_m"] = gap
            if gap < -epsilon:
                report.update(failure=phase + "_is_obstructed", failed_waypoint=index)
                return report

        start_xz = feet[[0, 2]]
        span = xz - start_xz
        intervals = max(1, math.ceil(np.linalg.norm(span) / min(0.05, radius / 2)))
        lowest, highest = sorted((y, next_floor))
        for fraction in np.arange(1, intervals) / intervals:
            if support(start_xz + fraction * span, lowest, highest) is None:
                report.update(failure="missing_support_between_waypoints", failed_waypoint=index)
                return report
        feet = landing
        report["feet_trace_m"].append(feet.tolist())

    report["passed"] = True
    return report


def testhall_traversal(triangles, *, check_supported_walk=False):
    """Replay the known TestHall opening and wall routes against supplied geometry."""
    routes = {
        "east_west_opening": (*route_endpoints(-4.27), CAPSULE_RADIUS, CAPSULE_HEIGHT, True),
        "open_passage": (*route_endpoints(-3.70), CAPSULE_RADIUS, CAPSULE_HEIGHT, True),
        "solid_wall": (*route_endpoints(-4.75), CAPSULE_RADIUS, CAPSULE_HEIGHT, False),
        "arch": ((11.72, 7.94, 0.20), (11.72, 7.94, -1.25), 0.28, 1.75, True),
    }
    results = {}
    for name, (start, end, radius, height, should_pass) in routes.items():
        gap = sweep_clearance(triangles, start, end, radius, height)
        passed = math.isfinite(gap) and (gap > 0.001 if should_pass else gap < -0.001)
        result = {"clearance_m": gap if math.isfinite(gap) else None, "expected_open": should_pass, "passed": passed}
        if check_supported_walk and should_pass and passed:
            first, last = np.asarray([start, end])[:, [0, 2]]
            n = math.ceil(np.linalg.norm(last - first) / 0.25)
            waypoints = np.linspace(first, last, n + 1)[1:]
            result["supported_walk"] = walk_route(triangles, start, waypoints, radius=radius, height=height)
            result["passed"] = result["supported_walk"]["passed"]
        results[name] = result
    ground = floor_height(triangles, 10.5, -3.70, 7.65, 8.15)
    floor_ok = ground is not None and abs(ground - 7.910168) <= 0.001
    return {"passed": floor_ok and all(route["passed"] for route in results.values()),
            "routes": results, "floor_y_m": ground, "floor_passed": floor_ok}


def staircase_fixture(*, rise=0.17, tread=0.32):
    """Return a four-tread render mesh, starting feet, and landing waypoints."""
    triangles, landings = [], []

    def quad(a, b, c, d):
        triangles.extend(((a, b, c), (a, c, d)))

    for step in range(4):
        y, start, end = step * rise, step * tread, (step + 1) * tread
        quad((-0.5, y, start), (-0.5, y, end), (0.5, y, end), (0.5, y, start))
        if step:
            low = (step - 1) * rise
            quad((-0.5, low, start), (-0.5, y, start), (0.5, y, start), (0.5, low, start))
        landings.append((0.0, start + tread / 4))

    x, z = landings[0]
    return np.asarray(triangles), np.array([x, 0.0, z]), np.asarray(landings[1:])
