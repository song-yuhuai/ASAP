#!/usr/bin/env python3
import argparse, os, time, math
import numpy as np
from isaacgym import gymapi  # Preview 4 API

def main():
    ap = argparse.ArgumentParser(description="URDF zero-state visualizer (Isaac Gym, P4) with joint probe/sweep.")
    ap.add_argument("--urdf", required=True, help="Path to the URDF file")
    ap.add_argument("--height", type=float, default=1.0, help="Spawn height for base (meters)")
    ap.add_argument("--fixbase", action="store_true", help="Fix base link")
    ap.add_argument("--probe_name", default=None, help="Joint name to rotate by --angle_deg")
    ap.add_argument("--probe_idx", type=int, default=None, help="Joint index to rotate by --angle_deg")
    ap.add_argument("--angle_deg", type=float, default=90.0, help="Angle for probe (degrees)")
    ap.add_argument("--sweep", action="store_true", help="Iterate through all DOFs, one per interval")
    ap.add_argument("--sweep_dt", type=float, default=1.0, help="Seconds per DOF when sweeping")
    args = ap.parse_args()

    urdf_path = os.path.abspath(args.urdf)
    asset_root = os.path.dirname(urdf_path)
    asset_file = os.path.basename(urdf_path)

    gym = gymapi.acquire_gym()

    # --- Sim (CPU pipeline keeps root-state tensors on CPU, matches your earlier setup)
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create sim")

    # --- Ground plane
    pp = gymapi.PlaneParams()
    pp.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, pp)

    # --- Viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer")

    # --- Env
    env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)

    # --- Asset
    aopts = gymapi.AssetOptions()
    aopts.fix_base_link = bool(args.fixbase)
    aopts.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)  # avoid deprecation warning
    print(f"[viz] Gym actor asset   : {asset_root} / {asset_file}")
    asset = gym.load_asset(sim, asset_root, asset_file, aopts)
    if asset is None:
        raise RuntimeError(f"Failed to load URDF asset: {urdf_path}")

    # --- Spawn
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, float(args.height))
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor = gym.create_actor(env, asset, pose, "robot", 0, 1)

    # --- DOF info
    dof_names = list(gym.get_asset_dof_names(asset))
    props = gym.get_asset_dof_properties(asset)
    lower = np.array(props["lower"], dtype=np.float32)
    upper = np.array(props["upper"], dtype=np.float32)

    print(f"[viz] actor DOFs={len(dof_names)}")
    for i, n in enumerate(dof_names):
        print(f"  {i:02d} {n:>28s}  limits=({lower[i]:+.3f}, {upper[i]:+.3f})")

    # --- Helpers
    def set_zero_states():
        st = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        st["pos"][:] = 0.0
        st["vel"][:] = 0.0
        gym.set_actor_dof_states(env, actor, st, gymapi.STATE_ALL)

    def apply_probe(dof_i: int, angle_rad: float):
        st = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        st["pos"][:] = 0.0
        st["pos"][dof_i] = float(np.clip(angle_rad, lower[dof_i], upper[dof_i]))
        st["vel"][:] = 0.0
        gym.set_actor_dof_states(env, actor, st, gymapi.STATE_ALL)

    # zero pose
    set_zero_states()

    # resolve which DOF to poke
    probe_idx = None
    if args.probe_name:
        if args.probe_name in dof_names:
            probe_idx = dof_names.index(args.probe_name)
        else:
            print(f"[warn] probe_name '{args.probe_name}' not found in URDF DOFs.")
    elif args.probe_idx is not None:
        if 0 <= args.probe_idx < len(dof_names):
            probe_idx = args.probe_idx
        else:
            print(f"[warn] probe_idx {args.probe_idx} out of range [0,{len(dof_names)-1}]")

    angle_rad = math.radians(args.angle_deg)

    # initial probe
    if args.sweep:
        i = 0
        apply_probe(i, angle_rad)
        print(f"[sweep] {i:02d} {dof_names[i]} ← {args.angle_deg:.1f} deg")
        next_switch = time.time() + args.sweep_dt
    elif probe_idx is not None:
        apply_probe(probe_idx, angle_rad)
        print(f"[probe] {probe_idx:02d} {dof_names[probe_idx]} ← {args.angle_deg:.1f} deg")
        next_switch = None
    else:
        next_switch = None

    # Camera
    gym.viewer_camera_look_at(
        viewer, env,
        gymapi.Vec3(3.0, -3.0, 2.0),
        gymapi.Vec3(0.0,  0.0, 1.0),
    )
    print("[info] Viewer controls: Right-drag orbit, Middle-drag pan, Scroll zoom. Close the window to exit.")

    # main loop
    while not gym.query_viewer_has_closed(viewer):
        if args.sweep and time.time() >= next_switch:
            i = (i + 1) % len(dof_names)
            apply_probe(i, angle_rad)
            print(f"[sweep] {i:02d} {dof_names[i]} ← {args.angle_deg:.1f} deg")
            next_switch = time.time() + args.sweep_dt

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()

