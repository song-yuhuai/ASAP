#!/usr/bin/env python3
"""
probe_dof_order.py — Check Isaac Gym asset DOF names/order vs. control list in YAML.

Usage:
  python probe_dof_order.py \
    --asset-root humanoidverse/resources/robots/x1 \
    --asset-file x1.urdf \
    --yaml humanoidverse/config/robots/x1.yaml \
    [--checkpoint PATH/TO/model_####.pt]  # optional; infers action_dim

What it does:
- Loads the robot asset via Isaac Gym (preferred) to obtain the *true* DOF order.
- Extracts a control list from your YAML (tries keys like motor_joints, actuated_joints, control_joints).
- Reports missing/extra names and order mismatches.
- Prints a corrected control list in the asset’s DOF order.
- Prints a permutation you can apply to actions if needed.
"""

import argparse
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- YAML (required) ---
try:
    import yaml
except Exception as e:
    print("ERROR: PyYAML not installed. Try:  pip install pyyaml", file=sys.stderr)
    raise

# --- Isaac Gym (preferred for true DOF order) ---
GYM_AVAILABLE = True
try:
    from isaacgym import gymapi
except Exception:
    GYM_AVAILABLE = False

# --- Optional: torch to infer action_dim from checkpoint ---
TORCH_AVAILABLE = True
try:
    import torch
except Exception:
    TORCH_AVAILABLE = False


def find_control_list(yaml_obj: Any) -> Optional[List[str]]:
    """
    Try to find a list of joint names that looks like the control set.
    We search keys containing 'motor', 'actuat', or 'control' and 'joint' (case-insensitive).
    Returns the first plausible list of strings.
    """
    results = []

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kp = k.lower()
                if isinstance(v, list):
                    # Consider this list if key name looks like motor/actuated/control joints
                    if re.search(r'(motor|actuat|control).*joint', kp):
                        if all(isinstance(x, str) for x in v) and len(v) > 0:
                            results.append((path + "/" + k if path else k, v))
                walk(v, path + "/" + k if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")

    walk(yaml_obj)

    # Heuristic: prefer keys that look most relevant by name
    def rank(k):
        k = k.lower()
        score = 0
        if "motor" in k: score += 3
        if "actuat" in k: score += 2
        if "control" in k: score += 1
        if "joint" in k: score += 2
        return -score  # lower is better

    if results:
        results.sort(key=lambda kv: (rank(kv[0]), len(kv[1])))
        return results[0][1]
    return None


def load_asset_dof_order(asset_root: Path, asset_file: str) -> List[str]:
    """
    Load the asset via Isaac Gym and return DOF names in the exact order Gym uses.
    """
    if not GYM_AVAILABLE:
        raise RuntimeError(
            "Isaac Gym not available in this environment. "
            "Install/activate Isaac Gym, or run inside the same env as your training."
        )

    g = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    # Minimal config; we just need to load an asset:
    sim = g.create_sim(0, 0, gymapi.SimType.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim (check GPU/driver/installation).")

    opts = gymapi.AssetOptions()
    asset = g.load_asset(sim, str(asset_root), asset_file, opts)
    if asset is None:
        raise RuntimeError(f"Failed to load asset: root={asset_root}, file={asset_file}")

    dof_names = g.get_asset_dof_names(asset)
    return list(dof_names)


def infer_action_dim_from_checkpoint(ckpt_path: Path) -> Optional[int]:
    if not TORCH_AVAILABLE:
        return None
    if not ckpt_path.exists():
        return None
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    except Exception:
        return None

    state = ckpt.get("state_dict", ckpt)

    def try_find(d: Dict[str, Any]) -> Optional[int]:
        for k, v in d.items():
            if hasattr(v, "ndim"):
                # common heads: bias is [act_dim], weight is [act_dim, hidden]
                if v.ndim == 1 and ("actor" in k or "mu.bias" in k or k.endswith("bias")):
                    return int(v.numel())
                if v.ndim == 2 and ("actor" in k or "mu.weight" in k or k.endswith("weight")):
                    return int(v.shape[0])
        return None

    act_dim = try_find(state)
    return act_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-root", required=True, type=Path)
    ap.add_argument("--asset-file", required=True, type=str)
    ap.add_argument("--yaml", required=True, type=Path, help="Robot config YAML containing control-joint list")
    ap.add_argument("--checkpoint", type=Path, default=None, help="(Optional) checkpoint to infer action_dim")
    ap.add_argument("--save-corrected", type=Path, default=None,
                    help="(Optional) write corrected control list (asset order) to this YAML file")
    args = ap.parse_args()

    # Load DOF order from asset (Gym)
    asset_dofs = load_asset_dof_order(args.asset_root, args.asset_file)

    # Load YAML and find control list
    y = yaml.safe_load(args.yaml.read_text())
    ctrl = find_control_list(y)

    print("\n=== Asset DOFs (Isaac Gym order) ===")
    for i, n in enumerate(asset_dofs):
        print(f"{i:02d}: {n}")
    print(f"Total asset DOFs: {len(asset_dofs)}")

    if ctrl is None:
        print("\nERROR: Could not find a control-joint list in YAML. "
              "Look for keys like 'motor_joints', 'actuated_joints', or 'control_joints'.", file=sys.stderr)
        sys.exit(2)

    print("\n=== Control joints from YAML ===")
    for i, n in enumerate(ctrl):
        print(f"{i:02d}: {n}")
    print(f"Total control joints: {len(ctrl)}")

    # Name set diffs
    missing_in_asset = [n for n in ctrl if n not in asset_dofs]
    extra_in_asset   = [n for n in asset_dofs if n not in ctrl]

    print("\n=== Name set comparison ===")
    print("In control but NOT in asset:", missing_in_asset or "OK")
    print("In asset but NOT in control:", extra_in_asset or "OK")

    # If sets match, check order and build permutation
    if not missing_in_asset and not extra_in_asset:
        order_mismatches = [(i, asset_dofs[i], ctrl[i]) for i in range(min(len(asset_dofs), len(ctrl)))
                            if asset_dofs[i] != ctrl[i]]
        if order_mismatches:
            print("\n=== ORDER MISMATCHES (index, asset_dof, control_dof) ===")
            for t in order_mismatches:
                print(t)
        else:
            print("\nOrder: OK")

        # Build permutation: given actions in CONTROL order, permute to ASSET order
        name_to_idx_ctrl = {n: i for i, n in enumerate(ctrl)}
        perm_ctrl_to_asset = [name_to_idx_ctrl[n] for n in asset_dofs]
        print("\nPermutation (control→asset):")
        print(perm_ctrl_to_asset)

        # Print corrected control list matching the asset order
        corrected = asset_dofs[:len(ctrl)]
        print("\n=== Corrected control list in ASSET order ===")
        for i, n in enumerate(corrected):
            print(f"{i:02d}: {n}")

        if args.save_corrected:
            # Write a tiny YAML with the corrected list (key name 'motor_joints' by default)
            out = {"motor_joints": corrected}
            args.save_corrected.write_text(yaml.safe_dump(out, sort_keys=False))
            print(f"\nSaved corrected control list to: {args.save_corrected}")

    else:
        print("\nFix the missing/extra names first. Order check is skipped until sets match exactly.")

    # Optional: checkpoint action dim
    if args.checkpoint:
        act_dim = infer_action_dim_from_checkpoint(args.checkpoint)
        print("\n=== Checkpoint action_dim (heuristic) ===")
        print(act_dim if act_dim is not None else "Could not infer (different model layout?)")


if __name__ == "__main__":
    main()
