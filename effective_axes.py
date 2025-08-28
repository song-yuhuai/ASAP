#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
effective_axes.py — Compute each joint's *effective axis in the parent frame*.

Supports:
  • URDF  (root <robot>)
  • MJCF/XML (MuJoCo, root <mujoco>)

What it does:
  1) Parses joint local axis + frame orientation.
  2) Rotates local axis into the parent frame:  eff_axis = R(parent←local) @ axis_local
  3) Prints a friendly table with:
       - joint name & type
       - local axis
       - frame orientation (URDF: origin rpy; MJCF: body quat/euler)
       - effective axis in parent frame
       - cosines vs the canonical axes (X=[1,0,0], Y=[0,1,0], Z=[0,0,1])
  4) Runs checks:
       - Waist triad orthogonality (lumber_yaw/roll/pitch)
       - Leg pitch-chain parallelism (hip_pitch ~ knee_pitch ~ ankle_pitch per side)

Notes:
  • URDF: <origin rpy="roll pitch yaw"> is the joint frame pose relative to the parent link.
          Default joint axis is [1, 0, 0] if <axis> is missing (per URDF spec).
  • MJCF: <joint ... axis="..."> is in the CHILD body frame. We use the body’s orientation
          (prefer <quat="w x y z">, fallback to <euler="x y z">) to rotate it into the parent.
  • Only revolute/continuous/prismatic (hinge/slide) joints are reported.

Usage:
  python effective_axes.py path/to/robot.urdf
  python effective_axes.py path/to/model.xml --focus "lumber|hip|knee|ankle" --warn_par=0.98 --warn_ort=0.15

Author: (you)
"""

import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, List, Dict

import numpy as np

# ---------- math helpers ----------

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    """URDF rpy = roll(x), pitch(y), yaw(z). Rotation parent←local = Rz(yaw) * Ry(pitch) * Rx(roll)"""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0],
                   [0,  cr, -sr],
                   [0,  sr,  cr]])
    return Rz @ Ry @ Rx

def quat_to_R(w: float, x: float, y: float, z: float) -> np.ndarray:
    """MuJoCo/MJCF: body quat is (w, x, y, z)."""
    q = np.array([w, x, y, z], dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def euler_xyz_to_R(ex: float, ey: float, ez: float, degrees: bool = True) -> np.ndarray:
    """Fallback for MJCF <euler='x y z'> (MuJoCo uses degrees, XYZ order)."""
    if degrees:
        ex, ey, ez = math.radians(ex), math.radians(ey), math.radians(ez)
    Rx = rpy_to_R(ex, 0, 0)
    Ry = rpy_to_R(0, ey, 0)
    Rz = rpy_to_R(0, 0, ez)
    return Rz @ Ry @ Rx  # xyz intrinsic ~ Z*Y*X (keeps consistency with vector rotation)

def cos_to_axes(v: np.ndarray) -> Tuple[float, float, float]:
    v = _norm(v)
    return float(abs(np.dot(v, X))), float(abs(np.dot(v, Y))), float(abs(np.dot(v, Z)))

# ---------- parsers ----------

def detect_format(path: str) -> str:
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        sys.exit(f"[FATAL] Could not parse XML: {e}")
    tag = re.sub(r"\{.*\}", "", root.tag).lower()
    if tag == "robot":
        return "urdf"
    if tag == "mujoco":
        return "mjcf"
    sys.exit(f"[FATAL] Unknown root tag <{tag}> — expected <robot> (URDF) or <mujoco> (MJCF).")

def parse_urdf(path: str) -> List[Dict]:
    root = ET.parse(path).getroot()
    out = []
    for j in root.iter("joint"):
        jtype = j.get("type", "").lower()
        if jtype not in ("revolute", "continuous", "prismatic"):
            continue
        name  = j.get("name", "?")
        origin= j.find("origin")
        rpy = [0.0, 0.0, 0.0]
        if origin is not None and origin.get("rpy"):
            rpy = [float(x) for x in origin.get("rpy").split()]
        axis_el = j.find("axis")
        # URDF default axis is [1,0,0] if missing
        axis = [1.0, 0.0, 0.0] if axis_el is None or not axis_el.get("xyz") \
               else [float(x) for x in axis_el.get("xyz").split()]
        R = rpy_to_R(*rpy)
        eff = _norm(R @ np.array(axis, float))
        out.append(dict(name=name, type=jtype, axis_local=axis, rpy=rpy, eff_axis=eff))
    return out

def parse_mjcf(path: str) -> List[Dict]:
    """Walk body tree; joints live inside bodies. For each joint, rotate axis by the body's orientation."""
    root = ET.parse(path).getroot()
    world_R = np.eye(3)

    def body_R(body: ET.Element) -> np.ndarray:
        if body.get("quat"):
            w, x, y, z = (float(v) for v in body.get("quat").split())
            return quat_to_R(w, x, y, z)
        if body.get("euler"):
            ex, ey, ez = (float(v) for v in body.get("euler").split())
            return euler_xyz_to_R(ex, ey, ez, degrees=True)
        return np.eye(3)

    out = []

    def walk(body: ET.Element, R_parent: np.ndarray):
        R_here = R_parent @ body_R(body)
        # joints defined in this body
        for j in body.findall("joint"):
            jtype = j.get("type", "").lower()  # hinge, slide, ball, etc.
            if jtype not in ("hinge", "slide"):
                continue
            name = j.get("name", "?")
            # MuJoCo default axis is [1,0,0] for hinge/slide if not provided
            axis = [1.0, 0.0, 0.0] if not j.get("axis") else [float(x) for x in j.get("axis").split()]
            eff = _norm(R_here @ np.array(axis, float))
            out.append(dict(name=name, type=jtype, axis_local=axis, quat_or_euler=True, eff_axis=eff))
        # recurse
        for child in body.findall("body"):
            walk(child, R_here)

    # start at worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        # some MJCF files put bodies directly under <mujoco>
        bodies = root.findall("body")
    else:
        bodies = worldbody.findall("body")

    for b in bodies:
        walk(b, world_R)
    return out

# ---------- reporting & checks ----------

import numpy as np
def pretty_vec(v) -> str:
    a = np.asarray(v, float).ravel()
    return "[" + " ".join(f"{x:+.2f}" for x in a) + "]"

def report(joints: List[Dict], focus_re: Optional[re.Pattern] = None):
    print("\n=== Effective Joint Axes (in parent frame) ===")
    print(f"{'idx':>3}  {'name':28s}  {'type':9s}  {'axis_local':17s}  {'effective':22s}  {'|cosX| |cosY| |cosZ|'}")
    rows = []
    for i, j in enumerate(joints):
        if focus_re and not focus_re.search(j["name"]):
            continue
        eff = j["eff_axis"]
        cx, cy, cz = cos_to_axes(eff)
        rows.append((i, j["name"], j["type"], j["axis_local"], eff, (cx, cy, cz)))

    for i, name, jtype, axis_local, eff, (cx, cy, cz) in rows:
        print(f"{i:3d}  {name:28s}  {jtype:9s}  {pretty_vec(axis_local):17s}  "
              f"{pretty_vec(eff):>22s}    {cx:5.2f}  {cy:5.2f}  {cz:5.2f}")
    return rows

def find(joints: List[Dict], name: str) -> Optional[np.ndarray]:
    for j in joints:
        if j["name"] == name:
            return j["eff_axis"]
    return None

def waist_check(joints: List[Dict], warn_parallel: float, warn_ortho: float):
    names = ["lumber_yaw_joint", "lumber_roll_joint", "lumber_pitch_joint"]
    vecs = [find(joints, n) for n in names]
    if any(v is None for v in vecs):
        return
    yaw, roll, pitch = vecs
    def c(a,b): return abs(float(np.dot(_norm(a), _norm(b))))
    print("\n--- Waist triad check (want orthogonal: yaw ⟂ roll ⟂ pitch) ---")
    print(f"|cos(yaw, roll)|  = {c(yaw, roll):.3f}")
    print(f"|cos(yaw, pitch)| = {c(yaw, pitch):.3f}")
    print(f"|cos(roll, pitch)|= {c(roll, pitch):.3f}")
    bad = False
    if c(yaw, roll) > warn_ortho:   bad = True; print("  ⚠ Not orthogonal: yaw vs roll")
    if c(yaw, pitch) > warn_ortho:  bad = True; print("  ⚠ Not orthogonal: yaw vs pitch")
    if c(roll, pitch) > warn_ortho: bad = True; print("  ⚠ Not orthogonal: roll vs pitch")
    if not bad: print("  ✓ Waist triad looks orthogonal.")

def leg_pitch_chain_check(joints: List[Dict], side: str, warn_parallel: float):
    hp = find(joints, f"{side}_hip_pitch_joint")
    kp = find(joints, f"{side}_knee_pitch_joint")
    ap = find(joints, f"{side}_ankle_pitch_joint")
    if hp is None or kp is None or ap is None:
        return
    def c(a,b): return abs(float(np.dot(_norm(a), _norm(b))))
    print(f"\n--- {side.capitalize()} leg pitch-chain parallelism (want parallel) ---")
    print(f"|cos(hip_pitch, knee_pitch)|   = {c(hp, kp):.3f}")
    print(f"|cos(hip_pitch, ankle_pitch)|  = {c(hp, ap):.3f}")
    print(f"|cos(knee_pitch, ankle_pitch)| = {c(kp, ap):.3f}")
    ok = True
    if c(hp, kp) < warn_parallel: ok = False; print("  ⚠ Not parallel: hip_pitch vs knee_pitch")
    if c(hp, ap) < warn_parallel: ok = False; print("  ⚠ Not parallel: hip_pitch vs ankle_pitch")
    if c(kp, ap) < warn_parallel: ok = False; print("  ⚠ Not parallel: knee_pitch vs ankle_pitch")
    if ok: print("  ✓ Pitch chain looks parallel.")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Compute effective joint axes (parent frame) for URDF/MJCF.")
    ap.add_argument("file", help="Path to URDF (.urdf) or MJCF (.xml)")
    ap.add_argument("--focus", default=None, help="Regex to filter joint names (e.g. 'lumber|hip|knee|ankle')")
    ap.add_argument("--warn_par", type=float, default=0.98, help="Warn if |cos| for parallel check < this (default 0.98)")
    ap.add_argument("--warn_ort", type=float, default=0.15, help="Warn if |cos| for orthogonality > this (default 0.15)")
    args = ap.parse_args()

    fmt = detect_format(args.file)
    print(f"\n[info] Detected format: {fmt.upper()} — {os.path.basename(args.file)}")

    if fmt == "urdf":
        joints = parse_urdf(args.file)
        print("[info] URDF default joint axis is [1,0,0] when <axis> is omitted.")
    else:
        joints = parse_mjcf(args.file)
        print("[info] MJCF joint axis is defined in the CHILD body frame; body orientation rotates it into parent.")

    focus_re = re.compile(args.focus) if args.focus else None
    rows = report(joints, focus_re)

    # Run built-in checks (if the relevant joints exist)
    waist_check(joints, warn_parallel=args.warn_par, warn_ortho=args.warn_ort)
    for side in ("left", "right"):
        leg_pitch_chain_check(joints, side, warn_parallel=args.warn_par)

    print("\nLegend:")
    print("  • axis_local : axis as written in the file (joint/local frame).")
    print("  • effective  : axis rotated into the parent frame (what the simulator actually uses).")
    print("  • |cosX|,|cosY|,|cosZ| : closeness to canonical axes. Values near 1.00 mean 'mostly along that axis'.")

if __name__ == "__main__":
    main()
