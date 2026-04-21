import argparse
import json

import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import IncludeCurrent, InitializeFromCommandline, RestrictToRepacking
from pyrosetta.rosetta.core.scoring import ScoreType
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.protocols.relax import FastRelax


def _collect_scores(pose, scorefxn):
    scorefxn(pose)
    return {str(key): float(value) for key, value in pose.scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdb")
    parser.add_argument("output_pdb")
    parser.add_argument("--scores-json")
    parser.add_argument("--coord-cst-weight", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument(
        "--mode",
        choices=[
            "score_only",
            "backbone_only",
            "sidechains_only",
            "coupled_backbone_sidechains",
        ],
        required=True,
    )
    parser.add_argument("--constrain-to-start", dest="constrain_to_start", action="store_true")
    parser.add_argument("--no-constrain-to-start", dest="constrain_to_start", action="store_false")
    parser.set_defaults(constrain_to_start=True)
    args = parser.parse_args()

    print("MN_STAGE init_pyrosetta", flush=True)
    pyrosetta.init("-corrections::beta_nov16 -detect_disulf false -run:preserve_header true")

    print("MN_STAGE load_pose", flush=True)
    pose = pose_from_pdb(args.input_pdb)
    print("MN_STAGE configure_relax", flush=True)
    scorefxn = pyrosetta.create_score_function("beta_nov16")
    scorefxn.set_weight(ScoreType.coordinate_constraint, float(args.coord_cst_weight))
    before_scores = _collect_scores(pose, scorefxn)

    def run_relax_step(*, allow_backbone: bool, allow_sidechains: bool, stage_name: str):
        movemap = MoveMap()
        movemap.set_bb(allow_backbone)
        movemap.set_chi(allow_sidechains)
        movemap.set_jump(False)

        task_factory = TaskFactory()
        task_factory.push_back(InitializeFromCommandline())
        task_factory.push_back(IncludeCurrent())
        task_factory.push_back(RestrictToRepacking())

        relax = FastRelax()
        relax.set_scorefxn(scorefxn)
        relax.set_movemap(movemap)
        relax.set_task_factory(task_factory)
        # Important: in FastRelax, sidechain repacking can still happen even when chi minimization is disabled.
        # This flag makes fixed-chi positions non-packable, so "backbone only" truly keeps sidechains fixed.
        relax.set_movemap_disables_packing_of_fixed_chi_positions(True)
        relax.constrain_relax_to_start_coords(bool(args.constrain_to_start))
        relax.max_iter(int(args.max_iter))
        print(f"MN_STAGE {stage_name}", flush=True)
        relax.apply(pose)

    def run_backbone_only_minimization():
        movemap = MoveMap()
        movemap.set_bb(True)
        movemap.set_chi(False)
        movemap.set_jump(False)

        minimizer = MinMover()
        minimizer.movemap(movemap)
        minimizer.score_function(scorefxn)
        minimizer.min_type("lbfgs_armijo_nonmonotone")
        minimizer.max_iter(int(args.max_iter))
        minimizer.tolerance(1e-4)
        minimizer.cartesian(False)
        print("MN_STAGE run_backbone_minimization_only", flush=True)
        minimizer.apply(pose)

    if args.mode == "score_only":
        print("MN_STAGE run_score_only", flush=True)
    elif args.mode == "backbone_only":
        run_backbone_only_minimization()
    elif args.mode == "sidechains_only":
        run_relax_step(allow_backbone=False, allow_sidechains=True, stage_name="run_fastrelax_sidechains_only")
    elif args.mode == "coupled_backbone_sidechains":
        run_relax_step(allow_backbone=True, allow_sidechains=True, stage_name="run_fastrelax_coupled")
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    print("MN_STAGE fastrelax_done", flush=True)
    after_scores = _collect_scores(pose, scorefxn)

    print("MN_STAGE write_output_pdb", flush=True)
    pose.dump_pdb(args.output_pdb)
    if args.scores_json:
        print("MN_STAGE write_scores", flush=True)
        with open(args.scores_json, "w", encoding="utf-8") as handle:
            json.dump({"before": before_scores, "after": after_scores}, handle, indent=2, sort_keys=True)
    print("MN_STAGE done", flush=True)


if __name__ == "__main__":
    main()
