# Taskboard
| Task ID | Priority | Status | Owner | Dependency | Worktree | Baseline | Lane Health | Summary | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | p0 | in-progress | omg-executor | - | root | HEAD | clean | Patch/Simplify POS biaffine layers for export | pending verify |
| T2 | p0 | todo | omg-executor | T1 | root | HEAD | clean | Patch/Simplify Depparse Biaffine/CRF for export | pending verify |
| T3 | p1 | todo | omg-verifier | T1, T2 | root | HEAD | clean | Rust ORT verification of patched POS/Depparse | pending verify |
| T4 | p2 | todo | omg-executor | - | root | HEAD | clean | Documentation of conversion workarounds | pending |
