#!/usr/bin/env python3
import argparse, sys, re
from pathlib import Path

import yaml
from isaacgym import gymapi

def load_asset_dofs(asset_root: Path, asset_file: str):
    g = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.use_gpu_pipeline = False
    sim = g.create_sim(0, 0, gymapi.SimType.SIM_PHYSX, sp)
    if sim is None:
        raise RuntimeError("Failed to create sim")
    opts = gymapi.AssetOptions()
    asset = g.load_asset(sim, str(asset_root), asset_file, opts)
    if asset is None:
        raise RuntimeError(f"Failed to load asset: {asset_root}/{asset_file}")
    return list(g.get_asset_dof_names(asset))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-root", required=True, type=Path)
    ap.add_argument("--asset-file", required=True, type=str)
    ap.add_argument("--yaml", required=True, type=Path)
    args = ap.parse_args()

    asset_dofs = load_asset_dofs(args.asset_root, args.asset_file)
    y_all = yaml.safe_load(args.yaml.read_text())
    # NEW: allow nested root (default 'robot')
    y = y_all.get("robot", y_all)

    ctrl = y.get("dof_names")

    if not ctrl or not all(isinstance(x,str) for x in ctrl):
        # fallback: scan for any list-of-strings likely to be joint names
        def candidates(node, path=""):
            out=[]
            if isinstance(node, dict):
                for k,v in node.items():
                    out+=candidates(v, f"{path}/{k}" if path else k)
            elif isinstance(node, list) and node and all(isinstance(x,str) for x in node):
                out.append((path,node))
            return out
        cands = candidates(y)
        print("ERROR: no 'dof_names' key found. Candidates:\n" +
              "\n".join([f"- {p} (len={len(lst)})" for p,lst in cands[:8]]), file=sys.stderr)
        sys.exit(2)

    print("\n=== Asset DOFs (URDF/Gym order) ===")
    for i,n in enumerate(asset_dofs): print(f"{i:02d}: {n}")
    print(f"Total asset DOFs: {len(asset_dofs)}")

    print("\n=== YAML dof_names (control order) ===")
    for i,n in enumerate(ctrl): print(f"{i:02d}: {n}")
    print(f"Total control DOFs: {len(ctrl)}")

    # 2) Name set check
    missing_in_asset = [n for n in ctrl if n not in asset_dofs]
    extra_in_asset   = [n for n in asset_dofs if n not in ctrl]
    print("\n=== Name set comparison ===")
    print("In control but NOT in asset:", missing_in_asset or "OK")
    print("In asset but NOT in control:", extra_in_asset or "OK")

    if missing_in_asset or extra_in_asset:
        print("\nFix name set first (typo / stale name).")
        sys.exit(1)

    # 3) Order check + permutation (control→asset)
    mismatches = [(i, asset_dofs[i], ctrl[i]) for i in range(len(asset_dofs)) if asset_dofs[i]!=ctrl[i]]
    if mismatches:
        print("\n=== ORDER MISMATCHES (idx, asset, control) ===")
        for t in mismatches: print(t)
    else:
        print("\nOrder: OK (control list matches asset order).")

    name2idx_ctrl = {n:i for i,n in enumerate(ctrl)}
    perm_ctrl_to_asset = [name2idx_ctrl[n] for n in asset_dofs]
    print("\nPermutation (control→asset):")
    print(perm_ctrl_to_asset)

    # 4) Sanity-check per-DOF arrays line up with dof_names length
    keys = [
        "dof_pos_lower_limit", "dof_pos_upper_limit",
        "dof_vel_limits", "default_stiffness", "default_damping",
        "dof_effort_limits"
    ]
    print("\n=== Per-DOF arrays alignment ===")
    ok=True
    for k in keys:
        arr = y.get(k)
        if arr is None: 
            continue
        if not isinstance(arr, list):
            print(f"- {k}: not a list?", type(arr)); ok=False; continue
        if len(arr)!=len(ctrl):
            print(f"- {k}: length {len(arr)} != len(dof_names) {len(ctrl)}  <<< MISMATCH")
            ok=False
        else:
            print(f"- {k}: length OK ({len(arr)})")
    if ok: print("All per-DOF array lengths match dof_names.")
    else: print("Fix lengths/order of per-DOF arrays to match dof_names (most common fall-over cause).")

if __name__ == "__main__":
    main()
