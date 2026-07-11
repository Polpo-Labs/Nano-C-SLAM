# python-nano-c-slam

Python reproduction + 2D simulation of **Nano-C-SLAM** (Ultra-Lightweight
Collaborative SLAM for Robot Swarms, IEEE Access 2025). Goal: run the
collaborative SLAM swarm in simulation (no hardware), stress-test how many
drones can map a scene before it stops being usable, and visualize the
reconstruction (including Foxglove). Full analysis and roadmap live in
[`../notes/nano-c-slam-diary.md`](../notes/nano-c-slam-diary.md).

## Engineering principles (binding — we are architects + PMs, not just coders)

1. **Modularity above all.** Every concern lives in its own component with a
   narrow interface, so we can test and debug each part in *isolation*. A
   component you can't run on its own is a component you can't debug.
2. **Readability is survival.** The day this turns into a wall of boilerplate is
   the day we can no longer debug it. Therefore:
   - Comment in plain language *what* we do and *what each variable means* —
     not just restating the code.
   - Keep it simple: less code is better code. Delete unnecessary complexity.
   - **DRY** — don't reinvent the wheel. Reuse and *adapt* an existing
     function/class rather than writing ten near-duplicates that differ by one
     line. Shared math lives in one place (`core/`).

## Layout

```
python-nano-c-slam/
├── nano_c_slam/          # the importable package
│   ├── core/             # shared foundations reused everywhere
│   │   ├── types.py      #   data structures: Pose2D, DepthFrame, AugmentedPose, Scan, Edge
│   │   └── geometry.py   #   SE(2) rigid-body math (poses <-> matrices, compose, relative)
│   ├── sim/              # the 2D world: kinematics, ToF ray-casting, noisy odometry, UWB
│   ├── slam/             # scan building, ICP, hierarchical PGO, distributed C-SLAM
│   ├── exploration/      # Cruise/Spinning/Caution state machine
│   ├── comms/            # token-based ranging + data protocol
│   └── viz/              # 2D monitor + Foxglove streaming
├── experiments/          # runnable scenarios + scalability studies
├── tests/                # pytest unit tests (one per isolated component)
├── requirements.txt
└── README.md
```

## Setup

Uses the repo-root virtual environment (`../.venv`). All Python runs through it:

```bash
../.venv/Scripts/python.exe -m pip install -r requirements.txt   # install deps
../.venv/Scripts/python.exe -m pytest                            # run tests
```
