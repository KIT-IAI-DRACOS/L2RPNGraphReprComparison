"""
Tooling to build reduced topology action spaces for the IEEE-36 and IEEE-118 grids using the
Fraunhofer `curriculumagent` N-1 Teacher method.

This package must be run in the separate `curriculum` conda environment (see
`environment_curriculum.yaml`), never in the main `L2RPN` training environment. It only
produces files consumed by the main pipeline (JSON action spaces under `data/action_spaces/`)
and never imports from `grid2op_env`.

Modules:
    line_selection: Pick per-grid `lines_to_attack` for the N-1 search from a cheap rollout.
    teacher_runner: Thin, corrected wrapper around `curriculumagent`'s `NMinusOneTeacher`.
    aggregation: Turn teacher_experience CSVs into a ranked, filtered action list.
    export: Convert ranked Grid2Op actions to this repo's action-space JSON format.
"""
