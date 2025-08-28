#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
axes_compare.py — Compare *effective joint axes (parent frame)* between two robot files.
Supports URDF (<robot>) and MuJoCo MJCF/XML (<mujoco>).

Usage:
  python axes_compare.py A.urdf B.xml --focus "lumber|hip|knee|ankle" --tol 0.98 --ort 0.15
  # Optional suggestions to make B match A (keeps B's frame rpy/quat and adjusts only <axis>):
  python axes_compare.py A.urdf B.xml --suggest fix=B --focus "hip|knee|ankle"

Outputs:
  - Joints present in both files: cosine similarity, angle (deg), and sign flag.
  - Missing-only-in-A / missing-only-in-B lists.
  - Waist orthogonality & leg pitch-chain parallel checks for each file.
  - If --suggest is set, prints <axis ...> lines to paste into the target file.
"""

import argparse, math, re, sys, xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import numpy as np

# ---------- math helpers ----------
def snap_vec_to_axis(v, eps=1e-6, max_deg=0.5):
    """
    If v is within max_deg of a canonical axis (±X, ±Y, ±Z), return that axis exactly.
    Otherwise return v normalized.
    """
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    v = v / n
    # candidates
    C = [np.array([+1,0,0]), np.array([-1,0,0]),
         np.array([0,+1,0]), np.array([0,-1,0]),
         np.array([0,0,+1]), np.array([0,0,-1])]
    best = max(C, key=lambda c: float(np.dot(v, c)))
    cosang = float(np.dot(v, best))
    ang_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    return best if ang_deg <= max_deg else v

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0],[sy, cy, 0],[0, 0, 1]])
    Ry = np.array([[cp, 0, sp],[0, 1, 0],[-sp, 0, cp]])
    Rx = np.array([[1, 0, 0],[0, cr, -sr],[0, sr, cr]])
    return Rz @ Ry @ Rx

def quat_to_R(w: float, x: float, y: float, z: float) -> np.ndarray:
    q = np.array([w, x, y, z], dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

# ---------- loaders returning effective axes and local rotations ----------
def detect_format(path: str) -> str:
    root = ET.parse(path).getroot()
    tag = re.sub(r"\{.*\}", "", root.tag).lower()
    if tag == "robot": return "urdf"
    if tag == "mujoco": return "mjcf"
    sys.exit(f"[FATAL] Unknown root tag <{tag}>")

def load_urdf(path: str, focus_re: Optional[re.Pattern]) -> Tuple[Dict[str,np.ndarray], Dict[str,np.ndarray]]:
    root = ET.parse(path).getroot()
    eff, Rloc = {}, {}
    for j in root.iter("joint"):
        jtype = j.get("type", "").lower()
        if jtype not in ("revolute","continuous","prismatic"): continue
        name = j.get("name", "?")
        if focus_re and not focus_re.search(name): continue
        origin = j.find("origin")
        rpy = [0.0,0.0,0.0]
        if origin is not None and origin.get("rpy"):
            rpy = [float(x) for x in origin.get("rpy").split()]
        axis_el = j.find("axis")
        axis = [1.0,0.0,0.0] if (axis_el is None or not axis_el.get("xyz")) \
               else [float(x) for x in axis_el.get("xyz").split()]
        R = rpy_to_R(*rpy)
        v = _norm(R @ np.array(axis, float))
        eff[name] = v
        Rloc[name] = R
    return eff, Rloc

def load_mjcf(path: str, focus_re: Optional[re.Pattern]) -> Tuple[Dict[str,np.ndarray], Dict[str,np.ndarray]]:
    root = ET.parse(path).getroot()
    eff, Rloc = {}, {}

    def body_R(body: ET.Element) -> np.ndarray:
        if body.get("quat"):
            w,x,y,z = (float(v) for v in body.get("quat").split())
            return quat_to_R(w,x,y,z)
        if body.get("euler"):
            ex,ey,ez = (float(v) for v in body.get("euler").split())
            # MuJoCo euler is XYZ in degrees
            Rx = rpy_to_R(math.radians(ex),0,0)
            Ry = rpy_to_R(0,math.radians(ey),0)
            Rz = rpy_to_R(0,0,math.radians(ez))
            return Rz @ Ry @ Rx
        return np.eye(3)

    def walk(body: ET.Element, R_parent: np.ndarray):
        R_here = R_parent @ body_R(body)
        for j in body.findall("joint"):
            jtype = j.get("type","").lower()
            if jtype not in ("hinge","slide"): continue
            name = j.get("name","?")
            if focus_re and not focus_re.search(name): continue
            axis = [1.0,0.0,0.0] if not j.get("axis") else [float(x) for x in j.get("axis").split()]
            v = _norm(R_here @ np.array(axis, float))
            eff[name] = v
            Rloc[name] = R_here  # parent<-local for this joint
        for child in body.findall("body"):
            walk(child, R_here)

    worldbody = root.find("worldbody")
    bodies = worldbody.findall("body") if worldbody is not None else root.findall("body")
    for b in bodies: walk(b, np.eye(3))
    return eff, Rloc

# ---------- compare & suggest ----------
def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(np.clip(np.dot(_norm(a), _norm(b)), -1.0, 1.0))
    return math.degrees(math.acos(abs(c)))  # ignore sign for angle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fileA")
    ap.add_argument("fileB")
    ap.add_argument("--focus", default=None, help="Regex to filter joints")
    ap.add_argument("--tol", type=float, default=0.98, help="parallel OK if |cos|>=tol (default 0.98)")
    ap.add_argument("--ort", type=float, default=0.15, help="orthogonal OK if |cos|<=ort (default 0.15)")
    ap.add_argument("--suggest", choices=["fix=A","fix=B"], default=None,
                    help="Print <axis ...> lines to edit the target file so it matches the other")
    args = ap.parse_args()
    focus_re = re.compile(args.focus) if args.focus else None

    fmtA = detect_format(args.fileA); fmtB = detect_format(args.fileB)
    if fmtA=="urdf": effA, RA = load_urdf(args.fileA, focus_re)
    else:            effA, RA = load_mjcf(args.fileA, focus_re)
    if fmtB=="urdf": effB, RB = load_urdf(args.fileB, focus_re)
    else:            effB, RB = load_mjcf(args.fileB, focus_re)

    namesA, namesB = set(effA), set(effB)
    both = sorted(namesA & namesB)
    onlyA = sorted(namesA - namesB)
    onlyB = sorted(namesB - namesA)

    print(f"\n[info] A={args.fileA} ({fmtA}), B={args.fileB} ({fmtB})")
    if onlyA: print("\nJoints only in A:", ", ".join(onlyA))
    if onlyB: print("\nJoints only in B:", ", ".join(onlyB))

    print("\n=== Axis match (effective, parent frame) ===")
    print(f"{'name':28s}  {'|cos|':>6s}  {'angle°':>7s}  {'sign':>6s}")
    bad = []
    for n in both:
        a, b = effA[n], effB[n]
        c = float(np.dot(_norm(a), _norm(b)))
        ang = angle_deg(a,b)
        sign = "same" if c >= 0 else "flip"
        print(f"{n:28s}  {abs(c):6.3f}  {ang:7.2f}  {sign:6s}")
        if abs(c) < args.tol:
            bad.append(n)
    if bad:
        print("\n⚠ Mismatch (not parallel enough):", ", ".join(bad))
    else:
        print("\n✓ All common joints are parallel (up to sign).")

    # waist & pitch-chain checks (quick) for each file individually
    def cos(a,b): return abs(float(np.dot(_norm(a), _norm(b))))
    def triad(eff):
        n = ["lumber_yaw_joint","lumber_roll_joint","lumber_pitch_joint"]
        if not all(k in eff for k in n): return
        y, r, p = eff[n[0]], eff[n[1]], eff[n[2]]
        print("\n[waist triad check]")
        print("|cos(yaw,roll)| =", f"{cos(y,r):.3f}",
              " |cos(yaw,pitch)| =", f"{cos(y,p):.3f}",
              " |cos(roll,pitch)| =", f"{cos(r,p):.3f}")
    def chain(eff, side):
        keys = [f"{side}_hip_pitch_joint", f"{side}_knee_pitch_joint", f"{side}_ankle_pitch_joint"]
        if not all(k in eff for k in keys): return
        hp,kp,ap = eff[keys[0]], eff[keys[1]], eff[keys[2]]
        print(f"[{side} pitch-chain] |cos(hp,kp)|={cos(hp,kp):.3f} |cos(hp,ap)|={cos(hp,ap):.3f} |cos(kp,ap)|={cos(kp,ap):.3f}")

    print("\n--- Sanity (A) ---"); triad(effA); chain(effA,"left"); chain(effA,"right")
    print("\n--- Sanity (B) ---"); triad(effB); chain(effB,"left"); chain(effB,"right")

    # Suggestions: compute local axis to set so target's effective matches source
    if args.suggest:
        target = "A" if args.suggest=="fix=A" else "B"
        print(f"\n=== Suggestions to make {target} match the other (change only <axis>, keep rpy/quat) ===")
        if target=="A":
            for n in both:
                R = RA.get(n); v = effB[n]
                if R is None: continue
                a_local = np.linalg.inv(R).dot(v); a_local = a_local/np.linalg.norm(a_local)
                a_local = snap_vec_to_axis(a_local, max_deg=0.5)  # snap near-canonical
                # pretty rounding for readability
                a_local = np.round(a_local, 6)
                if fmtA=="urdf":
                    print(f'{n:28s}  <-- set in A: <axis xyz="{a_local[0]:+.6f} {a_local[1]:+.6f} {a_local[2]:+.6f}"/>')
                else:
                    print(f'{n:28s}  <-- set in A: axis="{a_local[0]:+.6f} {a_local[1]:+.6f} {a_local[2]:+.6f}"')
        else:
            for n in both:
                R = RB.get(n); v = effA[n]
                if R is None: continue
                a_local = np.linalg.inv(R).dot(v); a_local = a_local/np.linalg.norm(a_local)
                a_local = snap_vec_to_axis(a_local, max_deg=0.5)  # snap near-canonical
                # pretty rounding for readability
                a_local = np.round(a_local, 6)
                if fmtB=="urdf":
                    print(f'{n:28s}  <-- set in B: <axis xyz="{a_local[0]:+.6f} {a_local[1]:+.6f} {a_local[2]:+.6f}"/>')
                else:
                    print(f'{n:28s}  <-- set in B: axis="{a_local[0]:+.6f} {a_local[1]:+.6f} {a_local[2]:+.6f}"')

if __name__ == "__main__":
    main()
