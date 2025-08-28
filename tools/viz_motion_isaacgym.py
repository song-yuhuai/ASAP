# tools/viz_motion_isaacgym.py

from isaacgym import gymapi, gymutil, gymtorch
import torch, os
import math
from types import SimpleNamespace

import hydra

from omegaconf import OmegaConf

# same library the env uses
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

def set_root_pose_cpu(gym, env, actor, pos_xyz, quat_xyzw):
    state = gymapi.ActorRootState()
    state.pose.p = gymapi.Vec3(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]))
    # Isaac Gym root expects (x,y,z,w). Start with xyzw; if it looks twisted, swap from wxyz → xyzw below.
    qx, qy, qz, qw = float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3])
    # If orientation looks wrong, UNCOMMENT the next line:
    # qx, qy, qz, qw = float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3]), float(quat_xyzw[0])
    state.pose.r = gymapi.Quat(qx, qy, qz, qw)
    state.vel = gymapi.Vec3(0.0, 0.0, 0.0)
    state.angvel = gymapi.Vec3(0.0, 0.0, 0.0)
    gym.set_actor_root_state(env, actor, state)


@hydra.main(version_base="1.1", config_path="../humanoidverse/config", config_name="base")
def main(cfg: OmegaConf):
    # 1) Build MotionLib exactly like training does (uses MJCF skeleton from cfg.robot.motion.asset.*)
    # pick dt exactly like training
    step_dt = float(cfg.env.dt) if "env" in cfg and "dt" in cfg.env else 0.02

    # define device BEFORE using it
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # make a non-struct copy so we can add step_dt safely
    base = OmegaConf.to_container(cfg.robot.motion, resolve=True)
    base["step_dt"] = step_dt
    motion_cfg = OmegaConf.create(base)

    # now initialize the motion lib exactly like training expects
    mlib = MotionLibRobot(motion_cfg, num_envs=1, device=device)
    mlib.load_motions(random_sample=False)


    motion_ids = torch.zeros(1, dtype=torch.long, device=device)
    t = torch.zeros(1, device=device)
    # wrap time by motion length (identical to training semantics)
    motion_len = mlib.get_motion_length(motion_ids).item()

    # 2) Isaac Gym sim (URDF actor from cfg.robot.asset.* — same as training)
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.dt = step_dt
    sim_params.use_gpu_pipeline = torch.cuda.is_available()
    sim_params.physx.use_gpu = torch.cuda.is_available()

    compute_id  = int(os.getenv("IGPU", "0"))       # or just 0
    graphics_id = int(os.getenv("VULKAN_DEVICE", "0"))  # or just 0
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False

    # sim setup
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    

    sim = gym.create_sim(compute_id, graphics_id, gymapi.SIM_PHYSX, sim_params)

    # ground plane
    pp = gymapi.PlaneParams()
    pp.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # Z-up
    gym.add_ground(sim, pp)
    
    if sim is None: raise RuntimeError("Failed to create sim")
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None: raise RuntimeError("Failed to create viewer")
    

    asset_opts = gymapi.AssetOptions()
    asset_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)

    asset_opts.collapse_fixed_joints = True
    asset = gym.load_asset(sim, cfg.robot.asset.asset_root, cfg.robot.asset.urdf_file, asset_opts)

    print("[viz] MotionLib skeleton:", motion_cfg.asset.assetRoot, "/", motion_cfg.asset.assetFileName)
    print("[viz] Gym actor asset   :", cfg.robot.asset.asset_root, "/", cfg.robot.asset.urdf_file)


    env = gym.create_env(sim, gymapi.Vec3(-1,0,0), gymapi.Vec3(1,1,1), 1)
    pose = gymapi.Transform()
    pose.p.z = 1.0
    actor = gym.create_actor(env, asset, pose, "robot", 0, 1)

    root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    gym.refresh_actor_root_state_tensor(sim)  # optional here, needed if you read current state
    print("[viz] root_state device:", root_state.device)  # should be cpu

    dof_props = gym.get_actor_dof_properties(env, actor)
    for i in range(dof_props.shape[0]):
        dof_props["stiffness"][i] = 80.0
        dof_props["damping"][i] = 4.0
    gym.set_actor_dof_properties(env, actor, dof_props)

    dof_count = gym.get_actor_dof_count(env, actor)
    print(f"[viz] actor DOFs={dof_count}")
    # (Optional) verify names match your robot_config if you want:
    # print(gym.get_asset_dof_names(asset))

    #root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(-1, 13)  # [pos(3), quat(4), linvel(3), angvel(3)]
    
    # ---------- URDF DOF order (Isaac Gym actor) ----------
    actor_dofs = list(gym.get_asset_dof_names(asset))

    # ---------- MJCF/MotionLib DOF order ----------
    mp = mlib.mesh_parsers  # Humanoid_Batch instance
    # DOF names in the EXACT order returned by motion_res["dof_pos"]
    ml_dofs = [ mp.mjcf_data['body_to_joint'][ mp.body_names[i] ] for i in mp.actuated_joints_idx ]

    def _norm(n: str) -> str:
        n = n.replace("motor_", "")
        n = n.replace("_joint", "")
        n = n.replace("lumber", "waist")   # common alias in X1
        return n

    actor_norm = [_norm(n) for n in actor_dofs]
    ml_norm    = [_norm(n) for n in ml_dofs]

    missing_from_ml = [n for n in actor_norm if n not in ml_norm]
    extra_in_ml     = [n for n in ml_norm if n not in actor_norm]

    print("[DBG dof] URDF dofs (actor):", actor_dofs, flush=True)
    print("[DBG dof] MJCF dofs (mlib) :", ml_dofs,    flush=True)
    print("[DBG dof] missing_from_ml :", missing_from_ml, flush=True)
    print("[DBG dof] extra_in_ml     :", extra_in_ml,     flush=True)

    # Proposed index map (print-only; DO NOT APPLY during this pass)
    perm_idx = [ ml_norm.index(n) if n in ml_norm else None for n in actor_norm ]
    print("[DBG dof] proposed (URDF -> MJCF indices):")
    for i, j in enumerate(perm_idx):
        right = (ml_dofs[j] if j is not None else "UNMAPPED")
        print(f"  {i:02d} {actor_dofs[i]:>24s}  <-  {j if j is not None else '??':>3}  {right}", flush=True)

    # Optional: axes from MJCF (can reveal sign/axis flips later)
    if hasattr(mp, "dof_axis"):
        print("[DBG dof] first few ML dof axes:", mp.dof_axis[:8], flush=True)



    frame = 0

    

    while not gym.query_viewer_has_closed(viewer):
        # advance time exactly like training (and wrap by motion length)
        t[:] = (frame * step_dt) % motion_len

        # offset = (0,0,0) origin; training uses env_origins — here it's the same
        motion_res = mlib.get_motion_state(motion_ids, t, offset=torch.zeros(1, 3, device=device))

        # root pose
        root_pos  = motion_res["root_pos"][0]
        root_quat = motion_res["root_rot"][0]   # try as-is; if twisted, reorder to xyzw

        # root_state has shape [num_actors, 13] = [pos(3), quat(4), linvel(3), angvel(3)]
        rs = root_state.view(-1, 13)

        # write position
        rs[0, 0:3] = root_pos.detach().cpu().float()

        # ---- QUATERNION HANDLING ----
        # Isaac Gym expects (x, y, z, w)
        # Many libs output either xyzw OR wxyz. We'll try xyzw first, then you can flip one flag.
        root_quat_t = root_quat.detach().cpu().float()

        # Normalize for safety (bad norms can cause wild rotations)
        root_quat_t = root_quat_t / torch.linalg.norm(root_quat_t).clamp_min(1e-8)

        USE_WXYZ_TO_XYZW = False  # set to True if the terrain/actor looks 90° off

        if USE_WXYZ_TO_XYZW:
            # convert from wxyz -> xyzw
            root_quat_t = torch.stack([root_quat_t[1], root_quat_t[2], root_quat_t[3], root_quat_t[0]])

        # write quaternion (xyzw as Gym expects)
        rs[0, 3:7] = root_quat_t

        # zero velocities (optional)
        rs[0, 7:13] = 0.0

        # push to sim
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))


        # joints
        q_ref = motion_res["dof_pos"][0]
        if q_ref.numel() != dof_count:
            print(f"[viz] DOF mismatch: ref={q_ref.numel()} actor={dof_count}")
            break

        

        

        #gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))

        
        if frame == 0: 
            st0 = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            q_act0 = torch.from_numpy(st0["pos"]).float()
            q_ref0 = motion_res["dof_pos"][0].detach().cpu().float()

            bias = q_ref0 - q_act0  # radians
            deg  = 180.0 / math.pi
            BIG  = (bias.abs() > 0.1).nonzero(as_tuple=False).flatten().tolist()

            print("[DBG zero] largest neutral offsets (>0.1 rad):")
            for i in BIG:
                print(f"  {i:02d} {actor_dofs[i]:>24s}  "
                    f"ref0={q_ref0[i]*deg:+7.2f}°  act0={q_act0[i]*deg:+7.2f}°  bias={bias[i]*deg:+7.2f}°")


        gym.set_actor_dof_position_targets(env, actor, q_ref.detach().cpu().numpy())



        gym.simulate(sim); gym.fetch_results(sim, True)
        gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        frame += 1

    gym.destroy_viewer(viewer); gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
