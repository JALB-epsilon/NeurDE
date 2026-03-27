# Training Scripts

Detached launchers for the requested SOD and Burgers runs live here.

Use `./training/start_todo_tmux.sh` to start the jobs in background `tmux` sessions.

Default GPU mapping:

- `GPU_SOD_CASE1=0`
- `GPU_SOD_CASE2_LEARNED=1`
- `GPU_SOD_CASE2_CONS_ANALYTIC=2`
- `GPU_SOD_CASE2_CONS_LEARNED=3`
- `GPU_BURGERS1D=4`

Override any of them before launch if your machine uses different device IDs.

Useful helpers:

- `./training/status_todo_tmux.sh`
- `./training/stop_todo_tmux.sh`

Logs are written to `training/logs/`.

For Burgers stage 2 after the stage-1 checkpoint exists:

- `./training/start_burgers_stage2_tmux.sh`

For queued Burgers sinusoidal data generation + stage 1 + stage 2:

- `./training/start_burgers_sinusoidal_tmux.sh`
