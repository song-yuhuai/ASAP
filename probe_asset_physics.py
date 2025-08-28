#!/usr/bin/env python3
"""
probe_asset_physics.py — Axis/PD/Inertia/Zero-Action checks for an Isaac Gym URDF + YAML.

Usage:
  python probe_asset_physics.py \
    --asset-root humanoidverse/data/robots/x1 \
    --asset-file xyber_x1_serial.urdf \
    --yaml humanoidverse/config/robot/x1/xyber_x1_serial.yaml \
    [--steps 240] [--dt 0.008333] [--substeps 2] [--poke lumber_yaw_joint]

Notes:
- Requires isaacgym (gymapi). For inertia checks, tries urdfpy; if missing, it skips C with a warning.
- The YAML is assumed to have a nested 'robot' section (robot.dof_names, gains, limits).
"""

import argparse, sys, re, math
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET
import math



import yaml

# --- Try to import urdfpy (for inertia checks) ---
HAVE_URDFPY = True
try:
    from urdfpy import URDF
except Exception:
    HAVE_URDFPY = False

# --- Isaac Gym ---
from isaacgym import gymapi

def load_yaml(path: Path):
    y_all = yaml.safe_load(path.read_text())
    y = y_all.get("robot", y_all)
    return y_all, y

def dump_axes(urdf_path: Path):
    txt = urdf_path.read_text()
    rows = []
    for m in re.finditer(r'<joint[^>]*name="([^"]+)"[^>]*type="([^"]+)"[^>]*>(.*?)</joint>', txt, re.S):
        name, jtype, body = m.groups()
        if jtype in ("revolute", "continuous", "prismatic"):
            axis = re.search(r'<axis[^>]*xyz="([^"]+)"', body)
            axis = (axis.group(1) if axis else "1 0 0").strip()
            rows.append((name, jtype, axis))
    print("\n=== A) Joint axes from URDF ===")
    for n, t, a in rows:
        print(f"{n:28s} type={t:10s} axis={a}")
    if not rows:
        print("(no revolute/continuous/prismatic joints found?)")
    return rows

def gym_load_asset(asset_root: Path, asset_file: str, dt: float, substeps: int):
    g = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.dt = dt
    sp.substeps = substeps
    sp.use_gpu_pipeline = False
    sp.physx.solver_type = 1  # TGS
    sim = g.create_sim(0, 0, gymapi.SimType.SIM_PHYSX, sp)
    if sim is None:
        raise RuntimeError("Failed to create sim (check GPU/driver/IsaacGym install)")
    opts = gymapi.AssetOptions()
    asset = g.load_asset(sim, str(asset_root), asset_file, opts)
    if asset is None:
        raise RuntimeError(f"Failed to load asset: {asset_root}/{asset_file}")
    return g, sim, asset

def dump_pd(g, asset):
    names = g.get_asset_dof_names(asset)
    props = g.get_asset_dof_properties(asset)
    print("\n=== B) DOF PD/limit/effort properties as applied by Gym ===")
    for i, n in enumerate(names):
        p = props[i]
        print(f"{i:02d} {n:28s} mode={p['driveMode']} "
              f"Kp={p['stiffness']:.1f} Kd={p['damping']:.1f} "
              f"vmax={p['velocity']:.2f} tmax={p['effort']:.2f} "
              f"low={p['lower']:.3f} high={p['upper']:.3f}")
    return names, props

def check_inertias(urdf_path: Path):
    print("\n=== C) Inertia/mass sanity ===")
    try:
        tree = ET.parse(str(urdf_path))
    except Exception as e:
        print(f"Could not parse URDF: {e}")
        return
    root = tree.getroot()
    ns = ""
    bad = False
    total_mass = 0.0
    for link in root.findall(f"{ns}link"):
        name = link.get("name", "?")
        inertial = link.find(f"{ns}inertial")
        if inertial is None:
            print("NO INERTIA:", name); bad = True; continue
        mass_el = inertial.find(f"{ns}mass")
        inertia_el = inertial.find(f"{ns}inertia")
        try:
            mass = float(mass_el.get("value"))
            total_mass += mass
        except Exception:
            print("BAD MASS:", name, mass_el.get("value") if mass_el is not None else None)
            bad = True
        try:
            ixx = float(inertia_el.get("ixx")); iyy = float(inertia_el.get("iyy")); izz = float(inertia_el.get("izz"))
            ixy = float(inertia_el.get("ixy")); ixz = float(inertia_el.get("ixz")); iyz = float(inertia_el.get("iyz"))
            # Cheap PD: diag must be positive-ish; not too ill-conditioned
            if min(ixx, iyy, izz) <= 1e-10:
                print("BAD INERTIA DIAGONAL:", name, (ixx, iyy, izz)); bad = True
        except Exception:
            print("MALFORMED INERTIA TAG:", name); bad = True
    print(f"Total mass (rough): {total_mass:.3f} kg")
    if not bad:
        print("Inertias look reasonable (no obvious degeneracies).")




def zero_action_stand_test(g, sim, asset, steps: int, poke_name: Optional[str] = None, yaml_robot=None):
    print("\n=== D) Zero-action stand test (headless) ===")

    # Ground + env + actor
    plane_params = gymapi.PlaneParams()
    g.add_ground(sim, plane_params)
    env = g.create_env(sim, gymapi.Vec3(-2,0,0), gymapi.Vec3(2,2,2), 1)
    pose = gymapi.Transform(); pose.p = gymapi.Vec3(0,0,1.0)
    actor = g.create_actor(env, asset, pose, "x1", 0, 1)

    # Names/props
    dof_names = list(g.get_asset_dof_names(asset))
    props = g.get_asset_dof_properties(asset)

    # --- Apply PD from YAML if available, else safe defaults ---
    Kp = yaml_robot.get("default_stiffness") if yaml_robot else None
    Kd = yaml_robot.get("default_damping")  if yaml_robot else None
    Eff= yaml_robot.get("dof_effort_limits") if yaml_robot else None
    Low= yaml_robot.get("dof_pos_lower_limit") if yaml_robot else None
    High= yaml_robot.get("dof_pos_upper_limit") if yaml_robot else None

    # Fallback safe values
    def safe_kp(name):
        return 25.0 if "lumber" in name else 80.0
    def safe_kd(name): return 3.0
    def safe_eff(name): return 60.0

    for i, n in enumerate(dof_names):
        props[i]["driveMode"] = gymapi.DOF_MODE_POS
        props[i]["stiffness"] = (Kp[i] if (isinstance(Kp, list) and i < len(Kp)) else safe_kp(n))
        props[i]["damping"]   = (Kd[i] if (isinstance(Kd, list) and i < len(Kd)) else safe_kd(n))
        if isinstance(Eff, list) and i < len(Eff): props[i]["effort"] = Eff[i]
        if isinstance(Low, list) and i < len(Low): props[i]["lower"]  = Low[i]
        if isinstance(High,list) and i < len(High):props[i]["upper"]  = High[i]

    g.set_actor_dof_properties(env, actor, props)

    # Targets: hold at 0 (clamped to limits). You can swap to nominal qpos if you have it.
    zeros = [0.0]*len(dof_names)
    # Optional poke to visualize axis sign/direction
    if poke_name and poke_name in dof_names:
        idx = dof_names.index(poke_name); zeros[idx] = 0.15
        print(f"(poke) +0.15 rad on {poke_name}")

    g.set_actor_dof_position_targets(env, actor, zeros)

    # Determine base body index (usually 0). We’ll print names for confidence.
    body_names = g.get_asset_rigid_body_names(asset)
    base_idx = 0
    print("Rigid bodies (asset order):", body_names[:4], "...")
    # Step
    z0 = None; zmin = math.inf
    for _ in range(steps):
        g.simulate(sim); g.fetch_results(sim, True)
        bodies = g.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
        try:
            z = bodies["pose"]["p"][base_idx]["z"]
        except Exception:
            # Some bindings return a flat numpy array in 13-float chunks
            b = bodies.reshape(-1, 13)
            z = b[base_idx, 2]  # x,y,**z**, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz
        z0 = z if z0 is None else z0
        zmin = min(zmin, z)
    drop = (z0 - zmin) if z0 is not None else float("nan")
    print(f"root_z start={z0:.3f} min={zmin:.3f} drop={drop:.3f}  (<=0.15 is stable-ish)")
    if drop > 0.3:
        print("⚠️  Large drop — unstable even with PD. Check joint axes, inertias, contact/friction.")


def rpy_to_R(r,p,y):
    import numpy as np
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz@Ry@Rx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-root", required=True, type=Path)
    ap.add_argument("--asset-file", required=True, type=str)
    ap.add_argument("--yaml", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--dt", type=float, default=1/120)
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--poke", type=str, default=None, help="Optional joint name to nudge +0.15 rad")
    args = ap.parse_args()

    y_all, y = load_yaml(args.yaml)
    urdf_path = args.asset_root / args.asset_file

    # A) axes
    dump_axes(urdf_path)

    # Gym load
    g, sim, asset = gym_load_asset(args.asset_root, args.asset_file, args.dt, args.substeps)

    # B) PD props
    names, props = dump_pd(g, asset)

    # Optional: compare with YAML dof_names length/order quickly
    dof_names = y.get("dof_names")
    if isinstance(dof_names, list) and all(isinstance(x,str) for x in dof_names):
        if dof_names != names:
            print("\n⚠️  YAML robot.dof_names differs from asset order — double-check mapping.")
    else:
        print("\n(Info) No robot.dof_names found at top-level under 'robot' — skipping mapping hint.")

    # C) inertia
    check_inertias(urdf_path)

    # D) zero-action stand test
    zero_action_stand_test(g, sim, asset, args.steps, args.poke, yaml_robot=y)

    tree=ET.parse("humanoidverse/data/robots/x1/xyber_x1_serial.urdf")
    for j in tree.getroot().iter("joint"):
        name=j.get("name"); jtype=j.get("type")
        if jtype not in ("revolute","continuous"): continue
        origin=j.find("origin"); rpy=[0,0,0]
        if origin is not None and origin.get("rpy"):
            rpy=[float(x) for x in origin.get("rpy").split()]
        axis=j.find("axis"); a=[0,0,1]
        if axis is not None and axis.get("xyz"):
            a=[float(x) for x in axis.get("xyz").split()]
        import numpy as np
        e = rpy_to_R(*rpy).dot(np.array(a))
        print(f"{name:28s} eff_axis_in_parent≈ [{e[0]:+.2f} {e[1]:+.2f} {e[2]:+.2f}]")


if __name__ == "__main__":
    main()
