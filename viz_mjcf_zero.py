#!/usr/bin/env python3
import argparse, math
try:
    import mujoco as mj
    from mujoco import viewer as mjv
except Exception as e:
    raise SystemExit(
        "This script requires the modern `mujoco` Python package (DeepMind MuJoCo ≥2.2).\n"
        "Install with: pip install mujoco\n"
        f"Import error: {e}"
    )

def set_zero_state(model: mj.MjModel, data: mj.MjData, height: float):
    """
    Set all joints to their canonical zero:
      - free joint: qpos = [1,0,0,0, 0,0,height]
      - ball joint: qpos = [1,0,0,0]
      - hinge/slide: qpos = 0
    Then call mj_forward() to update kinematics.
    """
    for j in range(model.njnt):
        adr = model.jnt_qposadr[j]
        jtype = model.jnt_type[j]
        if jtype == mj.mjtJoint.mjJNT_FREE:
            # qpos = [x,y,z, qw,qx,qy,qz]
            data.qpos[adr:adr+3]   = [0.0, 0.0, height]
            data.qpos[adr+3:adr+7] = [1.0, 0.0, 0.0, 0.0]
        elif jtype == mj.mjtJoint.mjJNT_BALL:
            data.qpos[adr:adr+4] = [1.0, 0.0, 0.0, 0.0]
        else:
            # hinge or slide
            data.qpos[adr] = 0.0
    mj.mj_forward(model, data)

def list_dofs(model: mj.MjModel):
    names = []
    for j in range(model.njnt):
        nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        jtype = model.jnt_type[j]
        adr = model.jnt_qposadr[j]
        dof = model.jnt_dofadr[j]
        # report human-friendly joint type
        jtype_str = {mj.mjtJoint.mjJNT_FREE:"free",
                     mj.mjtJoint.mjJNT_BALL:"ball",
                     mj.mjtJoint.mjJNT_SLIDE:"slide",
                     mj.mjtJoint.mjJNT_HINGE:"hinge"}.get(jtype, str(int(jtype)))
        names.append((j, nm, jtype_str, adr, dof))
    return names

def main():
    ap = argparse.ArgumentParser(description="Visualize MJCF zero state (no physics).")
    ap.add_argument("--xml", required=True, help="Path to MJCF XML")
    ap.add_argument("--height", type=float, default=1.0,
                    help="Spawn height for free root (meters). Default: 1.0")
    ap.add_argument("--print", action="store_true",
                    help="Print joint list and zero qpos in degrees (hinges) before viewing")
                    # --- new args ---
    ap.add_argument("--probe_name", default=None, help="Joint name to rotate (+angle_deg)")
    ap.add_argument("--probe_idx", type=int, default=None, help="Joint index to rotate")
    ap.add_argument("--angle_deg", type=float, default=90.0, help="Angle in degrees for probe")
    ap.add_argument("--sweep", action="store_true", help="Iterate through all hinge/slide joints")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.xml)
    data  = mj.MjData(model)

    # Put model into exact zero configuration
    set_zero_state(model, data, height=args.height)

    if args.print:
        print(f"[info] model={args.xml}")
        print(f"[info] njnt={model.njnt}, nq={model.nq}, nv={model.nv}")
        dofs = list_dofs(model)
        print("[info] joints:")
        for j, nm, jt, adr, dof in dofs:
            if jt == "hinge" or jt == "slide":
                val = data.qpos[adr]
                unit = "rad" if jt == "hinge" else "m"
                extra = f" ({math.degrees(val):+.2f} deg)" if jt == "hinge" else ""
                print(f"  {j:02d}  {nm:>28s}  {jt:>5s}  qpos[{adr:>2d}] = {val:+.6f} {unit}{extra}")
            elif jt == "ball":
                q = data.qpos[adr:adr+4]
                print(f"  {j:02d}  {nm:>28s}  {jt:>5s}  quat(wxyz) = {q}")
            else:  # free
                q = data.qpos[adr:adr+7]
                print(f"  {j:02d}  {nm:>28s}  {jt:>5s}  [w,x,y,z, px,py,pz] = {q}")

    # after set_zero_state(model, data, args.height):
    probe_rad = math.radians(args.angle_deg)

    def apply_probe(jid):
        jtype = model.jnt_type[jid]
        adr   = model.jnt_qposadr[jid]
        if jtype == mj.mjtJoint.mjJNT_HINGE or jtype == mj.mjtJoint.mjJNT_SLIDE:
            data.qpos[adr] = probe_rad if jtype == mj.mjtJoint.mjJNT_HINGE else probe_rad  # slides are meters; leave 0 if none
            mj.mj_forward(model, data)

    # resolve target joint id (if any)
    target_jid = None
    if args.probe_name is not None:
        target_jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, args.probe_name)
        if target_jid < 0:
            print(f"[warn] joint '{args.probe_name}' not found")
            target_jid = None
    elif args.probe_idx is not None:
        if 0 <= args.probe_idx < model.njnt:
            target_jid = args.probe_idx

    dt = 1/60.0                      # ~60 FPS target
    model.opt.timestep = dt          # optional, keeps cadence consistent
    # passive viewer loop
    with mjv.launch_passive(model, data) as viewer:
        # camera (as before) ...
        print("[info] Controls: right-drag orbit, middle-drag pan, scroll zoom. Close window to exit.")

        import time
        next_switch = time.time() + 1.0
        sweep_jids = [j for j in range(model.njnt)
                    if model.jnt_type[j] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE)]

        sweep_i = 0
        if args.sweep and sweep_jids:
            # start with zero, then apply first joint
            set_zero_state(model, data, args.height)
            apply_probe(sweep_jids[sweep_i])

        elif target_jid is not None:
            apply_probe(target_jid)

        while viewer.is_running():
            # on sweep, advance every second
            if args.sweep and time.time() >= next_switch:
                sweep_i = (sweep_i + 1) % len(sweep_jids)
                set_zero_state(model, data, args.height)
                apply_probe(sweep_jids[sweep_i])
                jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, sweep_jids[sweep_i])
                print(f"[sweep] {sweep_i:02d} {jname} ← {args.angle_deg:.1f} deg")
                next_switch = time.time() + 1.0
            t0 = time.perf_counter()
            viewer.sync()

            left = dt - (time.perf_counter() - t0)
            if left > 0:
                time.sleep(left)
            

if __name__ == "__main__":
    main()
