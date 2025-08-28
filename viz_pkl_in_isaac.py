#!/usr/bin/env python3
import argparse, pickle, time
import numpy as np
from isaacgym import gymapi

# put near the top of your script
try:
    import joblib
except Exception:
    joblib = None
import pickle, gzip, os, numpy as np

def load_any(path):
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)

def unwrap_single_container(x):
    # unwrap dict-with-single-item or single-element list/tuple until it's not
    while True:
        if isinstance(x, dict) and len(x) == 1:
            v = next(iter(x.values()))
            if isinstance(v, (dict, list, tuple)):
                x = v
                continue
        if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], dict):
            x = x[0]
            continue
        return x

def extract_motion(d):
    """
    Returns (names, qpos[T,D], meta)
    Supports keys:
      - qpos
      - qpos_rel + qpos_offset  (we add them)
      - dof  (uses .get('dof_names') if present; else assumes asset order)
    """
    meta = dict(d)  # shallow copy for extras like 'fps'
    names = d.get("dof_names") or d.get("names") or d.get("joint_names")

    if "qpos" in d:
        qpos = np.asarray(d["qpos"], np.float32)
        return names, qpos, meta

    if "qpos_rel" in d and "qpos_offset" in d:
        qpos = np.asarray(d["qpos_rel"], np.float32) + np.asarray(d["qpos_offset"], np.float32)
        return names, qpos, meta

    if "dof" in d:
        qpos = np.asarray(d["dof"], np.float32)
        # names may be missing here; we’ll fall back to assuming asset order
        return names, qpos, meta

    # If you ever want to support pose_aa → scalar per hinge, plug that in here.
    raise KeyError("No qpos / (qpos_rel+qpos_offset) / dof in file")



def load_motion(pkl_path):
    d = load_any(pkl_path)
    d = unwrap_single_container(d)
    names, qpos, meta = extract_motion(d)
    if not qpos.flags.writeable:
        qpos = qpos.copy()
    return names, qpos, meta



def get_asset_dof_names(gym, asset):
    n = gym.get_asset_dof_count(asset)
    return [gym.get_asset_dof_name(asset, i) for i in range(n)]

def build_perm(asset_names, motion_names):
    m_index = {n:i for i,n in enumerate(motion_names)}
    missing = [n for n in asset_names if n not in m_index]
    extra   = [n for n in motion_names if n not in set(asset_names)]
    if missing:
        print("[ERROR] Motion missing DOFs present in asset:", missing)
        raise SystemExit(1)
    if extra:
        print("[warn] Motion has extra DOFs not in asset (ignored):", extra)
    perm = np.array([m_index[n] for n in asset_names], dtype=np.int64)
    return perm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-root", required=True)
    ap.add_argument("--asset-file", required=True, help="URDF path relative to --asset-root")
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--dt", type=float, default=0.0, help="Simulation dt; if 0, use 1/fps from file when available")

    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--fix-base", action="store_true", help="Fix base link to quickly visualize")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--kp", type=float, default=300.0)
    ap.add_argument("--kd", type=float, default=10.0)
    args = ap.parse_args()

    # --- Load motion
    m_names, qpos, meta = load_motion(args.pkl)
    T, Dm = qpos.shape
    print(f"[info] motion: T={T} frames, D={Dm} DOFs from {args.pkl}")

    # If file has fps and the user didn't override dt, use it
    if "fps" in meta and (args.dt is None or args.dt <= 0):
        args.dt = 1.0 / float(meta["fps"])
        print(f"[info] using dt from fps: dt={args.dt:.6f}")

    

    # --- Gym init
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = args.dt
    sim_params.substeps = args.substeps
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.use_gpu = False

    sim = gym.create_sim(0, 0, gymapi.SimType.SIM_PHYSX, sim_params)
    if sim is None: raise SystemExit("[FATAL] create_sim failed")

    # --- Load URDF asset
    ao = gymapi.AssetOptions()
    ao.fix_base_link = args.fix_base
    ao.flip_visual_attachments = False
    ao.armature = 0.01
    ao.disable_gravity = False
    asset = gym.load_asset(sim, args.asset_root, args.asset_file, ao)
    if asset is None: raise SystemExit("[FATAL] load_asset failed")

    asset_names = get_asset_dof_names(gym, asset)
    Da = len(asset_names)
    print(f"[info] asset DOFs ({Da}): {asset_names}")

    # ... later, after loading the asset DOF names:
    

    # --- Build mapping (asset order <- motion order)
    if m_names is None:
        # No names in file; assume motion DOF order == asset DOF order if sizes match
        if Dm != Da:
            raise SystemExit(f"[ERROR] motion has {Dm} DOFs, asset has {Da}; "
                            "no names to map by. Provide a retargeted file with dof_names.")
        perm = np.arange(Da, dtype=np.int64)
        print("[warn] motion has no dof_names; assuming order matches asset order.")
        print(f"[info] example: asset[0]={asset_names[0]} <= motion[0] (unknown name)")
    else:
        perm = build_perm(asset_names, m_names)
        print("[info] name mapping OK. Example:",
            f"asset[0]={asset_names[0]} <= motion[{perm[0]}]={m_names[perm[0]]}")

    # --- Create env & actor
    env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 1.0), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # lift off ground a bit if base not fixed
    actor = gym.create_actor(env, asset, pose, "robot", 0, 0)

    # --- Drive mode: position control
    dof_props = gym.get_asset_dof_properties(asset)
    from isaacgym import gymutil  # only for constants
    # In Isaac Gym, driveMode constants live on gymapi (but gymutil makes it easy to access)
    for i in range(Da):
        dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
        dof_props["stiffness"][i] = args.kp
        dof_props["damping"][i]   = args.kd
    gym.set_actor_dof_properties(env, actor, dof_props)

    # --- Initial to frame 0 (mapped to asset order)
    q0_asset = qpos[0, perm].copy()
    dof_state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    dof_state["pos"][:Da] = q0_asset
    dof_state["vel"][:Da] = 0.0
    gym.set_actor_dof_states(env, actor, dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, actor, q0_asset)
    print("[info] set initial pose to motion frame 0")

    # --- Viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None: raise SystemExit("[FATAL] create_viewer failed")
    cam_pos = gymapi.Vec3(3.0, 0.0, 1.5); cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

    # --- timing / playback setup (put this after you compute T, load meta, etc.)
    fps = float(meta.get("fps", 30))
    if args.dt is None or args.dt <= 0:
        args.dt = 1.0 / fps
    steps_per_frame = max(1, int(round((1.0 / fps) / args.dt)))
    print(f"[info] fps={fps:.1f}, dt={args.dt:.6f}, steps_per_frame={steps_per_frame}")

    # (optional but harmless)
    gym.prepare_sim(sim)

    # --- main loop (replace your viewer loop with this)
    step_idx = 0
    t = 0
    print("[info] playing…  ESC to quit")

    while not gym.query_viewer_has_closed(viewer):
        # Update DOF targets on frame boundaries
        if step_idx % steps_per_frame == 0:
            targets = qpos[t, perm]
            gym.set_actor_dof_position_targets(env, actor, targets)
            t = (t + 1) if t + 1 < T else (0 if args.loop else T - 1)

        # Old Preview-4 stepping
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Render (only if viewer exists)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        step_idx += 1

    # on exit
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
