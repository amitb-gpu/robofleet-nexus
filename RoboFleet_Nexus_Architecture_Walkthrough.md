# RoboFleet Nexus — Architecture Walkthrough

> A layer-by-layer guide to the platform diagram, data flows, and component responsibilities.

---

## Reading the Diagram

The architecture is organized into five horizontal layers, read from bottom to top — hardware at the base, user-facing presentation at the top. Data flows upward through ingestion and processing, then fans out to the dashboard via WebSocket. Dashed-border boxes indicate planned Phase 2 / Phase 3 components not yet implemented.

---

## Layer 1 — Hardware Layer (bottom)

This layer represents physical and external resources that RoboFleet Nexus connects to but does not own.

### RTX A1000 6GB · Laptop GPU · WSL2
The primary development and deployment GPU. Runs the Nexus API server inside a conda environment on WSL2 (Ubuntu 24.04). Classified automatically as `WORKSTATION_DEV` based on its 6 GB VRAM. Capable of light Isaac Sim workloads and all diagnostic / RCA functions.

### RTX 5080 16GB · Laptop GPU · WSL2
A second GPU node on a separate machine on the same network. Classified as `WORKSTATION_HIGH` (16 GB VRAM). Intended to receive heavier Isaac Sim simulation jobs dispatched by the GPU-aware scheduler. Connected to the central Nexus instance via the Remote GPU Agent. Currently offline pending WiFi isolation fix on the local router.

### Physical Robot · ROS2 Jazzy · Ubuntu 24.04
Any ROS2-compatible robot running Ubuntu 24.04. Publishes standard ROS2 topics that the bridge subscribes to. The platform is robot-agnostic — any device publishing to `/battery_state`, `/diagnostics`, `/joint_states`, or `/odom` is supported without code changes.

### Isaac Sim / Gazebo · GPU-accelerated sim
NVIDIA Isaac Sim or Gazebo simulation environments. Simulation jobs are submitted via `POST /simulations/plan`, which uses the GPU scheduler to match workloads to available hardware profiles. Isaac Lab training runs (RL) are planned for Phase 2 experiment tracking.

### Claude API · claude-sonnet-4
Anthropic's hosted LLM API. Called asynchronously by the RCA agent whenever diagnostic findings are generated. Receives structured telemetry context and returns a JSON root-cause analysis. External to the platform — requires an API key and credits at `console.anthropic.com`.

### nvidia-smi · GPU metrics · poll 3s
The NVIDIA System Management Interface command-line tool. Called every 3 seconds by the background GPU monitor via `subprocess.run()`. Works identically in WSL2 and bare Linux. Returns VRAM usage, GPU utilization, temperature, power draw, and driver version for each detected device.

---

## Layer 2 — Ingestion Layer

This layer is responsible for getting data into the Nexus API. All components in this layer ultimately call `POST /telemetry` on the API server.

### ros2_bridge.py · rclpy · system Python 3.12
The real ROS2 bridge. Runs as a separate process using the system Python 3.12 installation (not the conda environment) because ROS2 Jazzy's C extension `.so` files are compiled for Python 3.12. The bridge initialises an `rclpy` node called `nexus_bridge` and subscribes to four topics:

| Topic | Message Type | What it captures |
|---|---|---|
| `/battery_state` | `sensor_msgs/BatteryState` | Voltage, current, charge percentage, power supply status |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Named hardware status entries with key-value diagnostic data |
| `/joint_states` | `sensor_msgs/JointState` | Position, velocity, and effort for each named joint |
| `/odom` | `nav_msgs/Odometry` | Robot pose (x, y, z, quaternion) and linear/angular velocity |

Each ROS2 callback converts the message into a `RobotEvent`-compatible dict using `_make_event()` and places it on a thread-safe `queue.Queue`. A separate flush thread (synchronous `httpx.Client`) drains the queue and POSTs events to `/telemetry`. This two-thread design decouples the rclpy spin loop from network I/O.

**WSL note:** Run with `PYTHONPATH=/path/to/robofleet-nexus/src` and source `/opt/ros/jazzy/setup.bash` before launching.

### mock_bridge.py · Dev / CI mode
A pure-Python telemetry simulator requiring no ROS2 installation. Publishes three event types on a configurable interval:

- **Battery drain** — starts at 100%, drains 0.15% per tick, triggers `WARNING` at 25% and `CRITICAL` at 15%
- **Joint states** — six joints with sinusoidal position and velocity values
- **Odometry** — circular path with radius 2m, advancing by a fixed angle each tick

Used for development, CI pipelines, and demonstrating the dashboard without physical hardware. Run with `--mock` flag.

### remote_gpu_agent.py · RTX 5080 node
A lightweight agent that runs on any remote GPU node and pushes that machine's `nvidia-smi` metrics to the central Nexus instance over HTTP (`POST /gpu/remote-snapshot`). Designed for the RTX 5080 laptop — once the network isolation issue is resolved, both GPUs will appear side-by-side in the dashboard GPU inventory panel.

### isaac/scheduler.py · GPU-aware job planner
Receives `IsaacSimulationJob` requests via `POST /simulations/plan` and matches them to available `SimulationProfile` entries based on VRAM requirements and device classification. Returns a `SimulationPlan` describing which GPU node should run the job and any constraints. Currently returns deterministic plans based on the current GPU inventory snapshot.

### gpu_monitor.py · Background asyncio task
Runs as a persistent asyncio task launched at server startup via the FastAPI lifespan context manager. Every 3 seconds it calls `nvidia-smi` via subprocess, parses the CSV output, classifies each device, and broadcasts a `gpu_snapshot` event to all connected WebSocket clients. Handles `[N/A]` values for laptop GPUs (e.g. power limit not reported).

**Device classification by VRAM:**

| Profile | VRAM threshold | Typical device |
|---|---|---|
| `LAPTOP_DEV` | ≤ 6.5 GB | RTX A1000 6GB |
| `WORKSTATION_DEV` | ≤ 10.5 GB | RTX 4060 |
| `WORKSTATION_HIGH` | ≤ 20 GB | RTX 5080 16GB |
| `PRODUCTION` | > 20 GB | RTX 6000 Ada, A100 |

### audit/hash_chain.py · SHA-256 tamper-evident log
Every significant event — telemetry ingest, diagnostic finding, simulation plan, GPU inventory check — is appended to an in-memory hash chain. Each record stores a SHA-256 hash of its own content plus the previous record's hash, forming a chain where any modification to a historical record is detectable. Accessible via `GET /audit`.

---

## Layer 3 — API Core Layer

The central FastAPI service. All external clients (bridges, browsers, curl) interact with this layer. Runs on `uvicorn` at `0.0.0.0:8000`.

### FastAPI · main.py
The main application entry point. Registers all routes, the WebSocket router, and the lifespan context manager (which starts the GPU poll loop). Key endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/telemetry` | Ingest a `RobotEvent`. Runs diagnostic rules, appends audit, triggers RCA if findings exist. |
| `GET` | `/diagnostics` | Return all active `DiagnosticFinding` objects. |
| `GET` | `/audit` | Return the full hash-chain audit log. |
| `GET` | `/gpu/inventory` | Query current GPU inventory from `nvidia-smi`. |
| `POST` | `/simulations/plan` | Submit an Isaac Sim job for GPU-aware scheduling. |
| `POST` | `/rca/analyze` | Manually trigger RCA for a robot against its current findings. |
| `GET` | `/rca` | Return all stored RCA results. |
| `GET` | `/dashboard` | Serve the live dashboard HTML. |
| `WS` | `/ws` | WebSocket endpoint for dashboard clients. |

### telemetry/schemas.py · Pydantic v2 models
All data contracts between platform components. Key models:

- **`RobotEvent`** — the universal telemetry payload. Fields: `event_id`, `robot_id`, `source` (ros2 / isaac_sim / mock / manual), `timestamp`, `severity`, `event_type`, `subsystem`, `message`, `metrics` (dict of floats), `metadata` (dict of any).
- **`DiagnosticFinding`** — a structured finding produced by the rules engine. Fields: `finding_id`, `robot_id`, `severity`, `title`, `explanation`, `evidence` (list of strings), `recommended_actions`, `requires_human_approval`.
- **`RobotSeverity`** — enum: `info`, `warning`, `critical`.
- **`GpuInventory`** / **`SimulationPlan`** / **`IsaacSimulationJob`** — GPU and simulation data contracts.

### diagnostics/rules.py · evaluate_event()
A pure function called synchronously on every telemetry ingest. Inspects the `metrics` dict of the incoming `RobotEvent` and generates zero or more `DiagnosticFinding` objects. Current rules:

| Metric | Threshold | Severity | Title |
|---|---|---|---|
| `battery_pct` | ≤ 15% | critical | Battery critically low |
| `battery_pct` | ≤ 25% | warning | Battery low |
| `motor_temp_c` | ≥ 90°C | critical | Motor thermal risk |
| `gpu_temp_c` | ≥ 85°C | warning | Elevated GPU temperature |
| `packet_loss_pct` | ≥ 5% | warning | Network degradation detected |
| `<joint>.eff` | ≥ 40 Nm | warning | High joint effort: `<joint>` |

New rules can be added by extending this file — no other changes required.

### ws_manager.py · ConnectionManager
Manages all active WebSocket connections. Key behaviours:

- **Replay buffer** — stores the last 50 events. New clients receive this history immediately on connect, so the dashboard is populated instantly rather than waiting for the next event cycle.
- **Fan-out broadcast** — `broadcast()` sends a typed event dict to all connected clients simultaneously.
- **Dead connection pruning** — clients that disconnect mid-send are silently removed.
- **Typed emitters** — `emit_telemetry()`, `emit_gpu_snapshot()`, `emit_diagnostic()`, `emit_rca_result()`, `emit_audit()`, `emit_ros2_event()`.

### ws_routes.py · FastAPI lifespan + /ws + /dashboard
Registers the WebSocket endpoint and dashboard HTML route. The `lifespan` async context manager starts the GPU poll background task on server startup and cancels it cleanly on shutdown. The `/ws` endpoint accepts connections, replays recent history, and keeps the connection alive.

---

## Layer 4 — Intelligence Layer

The AI reasoning layer. Triggered automatically when diagnostic findings are generated, runs asynchronously in the background so it never blocks the telemetry response.

### rca/agent.py · AI Root Cause Analysis Engine
The core of the platform's intelligence. When `evaluate_event()` returns one or more findings, `main.py` adds `_run_and_broadcast_rca()` as a FastAPI `BackgroundTask`. This function:

1. Collects all `RobotEvent` objects for the affected robot from the in-memory store (up to the last 10, most recent first)
2. Serialises the findings and event context to JSON
3. Calls `AsyncAnthropic.messages.create()` with the structured system prompt and user payload
4. Parses the JSON response
5. Appends `rca_id`, `robot_id`, `timestamp`, `findings_analyzed`, `input_tokens`, `output_tokens`
6. Stores the result and broadcasts it via WebSocket

**System prompt design:** The model is instructed to act as an expert robotics diagnostics engineer, respond only in structured JSON, prioritise safety (flag any risk of hardware damage or injury explicitly), and reference actual metric values from the evidence. It is told never to add markdown fences or preamble.

**Output schema:**
```
summary            — one sentence describing the overall situation
root_causes[]      — ranked list: cause, confidence, subsystem, evidence[]
recommended_actions[] — priority (immediate/soon/monitor), action, rationale
requires_human_approval — always true for critical findings
risk_level         — critical / high / medium / low
estimated_resolution_time
rca_id / robot_id / timestamp / findings_analyzed / input_tokens / output_tokens
```

**Fallback mode:** If the API key is missing or credits are exhausted, the agent returns a deterministic mock response built from the first finding's data. This ensures the dashboard always renders an RCA card and the pipeline never crashes on API errors.

### System Prompt Context box
Highlights the four key design decisions in the prompt: robotics engineer persona (domain-specific reasoning), JSON-only output (no parsing ambiguity), safety-first guidance (hardware damage and injury are flagged explicitly), and token usage tracking (cost visibility per call).

### RCA Results store · GET /rca
In-memory `list[dict]` accumulating all RCA results for the session. Accessible via REST for external consumers. In a production deployment this would be persisted to a database and indexed by `robot_id` and `timestamp`.

### BackgroundTask · FastAPI async
FastAPI's `BackgroundTasks` mechanism runs `_run_and_broadcast_rca()` after the HTTP response for `POST /telemetry` has already been sent. This means the telemetry ingest endpoint returns immediately (typically < 5ms) while the LLM call (typically 1-3 seconds) happens in the background without blocking the caller.

---

## Layer 5 — Presentation Layer (top)

Everything the user sees and interacts with.

### dashboard/index.html · Live Fleet Dashboard
A single HTML file served at `GET /dashboard`. No npm, no build step, no external dependencies beyond Google Fonts. Opens a WebSocket connection to `/ws` on load and renders all incoming events in real time.

**Panels:**

| Panel | Data source | Update trigger |
|---|---|---|
| KPI chips (robots, GPUs, jobs, diagnostics) | Derived from state | Every telemetry / GPU event |
| GPU inventory cards | `gpu_snapshot` events | Every 3 seconds |
| Robot fleet table | `telemetry` events | Every telemetry event |
| Live event stream | All event types | Every event |
| Diagnostic findings + AI RCA | `diagnostic` + `rca_result` events | When findings fire |
| Simulation job queue | `sim_job` events | On job status change |
| Audit log | `audit` events | On every audit append |

**Connection resilience:** The dashboard reconnects automatically after a 3-second delay if the WebSocket drops. On reconnect, the replay buffer ensures recent history is restored without requiring a full page refresh.

**RCA card rendering:** Risk level determines border and badge colour — red for critical, amber for high, blue for medium, green for low. Root causes are listed with rank and confidence. Immediate-priority actions are highlighted in red. Token usage is shown for cost transparency.

### Browser Client
Any modern browser connecting to `http://localhost:8000/dashboard`. The WebSocket connection is initiated by the dashboard JS on page load. The 50-event replay buffer means the dashboard is fully populated within milliseconds even if the server has been running for hours before the browser connects.

### Prometheus / Grafana *(Phase 2 — planned)*
A `GET /metrics` endpoint exposing Prometheus-format gauges and counters: GPU VRAM usage, active simulation job count, diagnostic rule hit rates, telemetry ingestion throughput. Will include a sample Grafana dashboard JSON for zero-friction adoption by robotics infrastructure teams already running the Prometheus / Grafana stack.

### Kubernetes Operator *(Phase 3 — planned)*
A Kopf-based Kubernetes operator defining two Custom Resource Definitions: `SimulationJob` (triggers GPU provisioning and job dispatch) and `RobotFleet` (spins up isolated Nexus instances per team). Enables one-command enterprise deployment via a Helm chart. Integrates with the NVIDIA GPU Operator for automatic node labelling.

### Quantum Router *(Phase 3 — planned)*
A hybrid classical-quantum module encoding multi-robot task allocation as a QUBO (Quadratic Unconstrained Binary Optimisation) matrix and solving it with a QAOA (Quantum Approximate Optimisation Algorithm) circuit implemented in Cirq and TensorFlow Quantum. GPU-accelerated state vector simulation via NVIDIA cuQuantum on the RTX 4060 / RTX 5080. Provides a genuinely differentiating technical narrative around NP-hard fleet routing problems.

---

## End-to-End Data Flow Walkthrough

Here is the complete journey of a single battery warning event through the platform:

```
1. Physical robot publishes /battery_state at 14% charge via ROS2
2. ros2_bridge.py callback fires: _battery_to_event() converts to RobotEvent dict
3. Event placed on thread-safe queue.Queue
4. Flush thread dequeues and POSTs to POST /telemetry (httpx.Client sync)
5. FastAPI ingest_telemetry() receives the RobotEvent
6. Event appended to EVENTS list and AUDIT_LOG hash chain
7. evaluate_event() checks battery_pct=14.0 → fires DiagnosticFinding (critical)
8. Finding appended to FINDINGS list and AUDIT_LOG
9. on_telemetry_ingested() broadcasts telemetry event to all WebSocket clients
10. Dashboard robot fleet table updates: bot_001 shows 14% battery, status=warn
11. FastAPI adds _run_and_broadcast_rca() as BackgroundTask — response returns 200 OK
12. BackgroundTask calls run_rca(robot_id, [finding], recent_events)
13. Last 10 RobotEvents for bot_001 retrieved (battery, joints, odometry history)
14. Structured payload sent to Claude API (claude-sonnet-4)
15. Claude returns JSON: risk=critical, root_cause=battery_discharge, immediate actions
16. RCA result stored in RCA_RESULTS list
17. WebSocket broadcast: rca_result event to all dashboard clients
18. Dashboard diagnostic findings panel renders red-bordered AI RCA card
19. Operator sees: ranked root causes, confidence, robot position, immediate actions
```

Total latency from ROS2 publish to dashboard update: ~50ms for telemetry card, ~1-3s for RCA card (Claude API response time).

---

## Key Design Decisions

**Why two Python environments?**
ROS2 Jazzy's C extensions are compiled for system Python 3.12. The Nexus conda environment runs Python 3.11. Running the bridge with `/usr/bin/python3` and setting `PYTHONPATH` to the project's `src/` directory allows both to coexist without conda conflicts.

**Why a synchronous httpx.Client in the bridge flush thread?**
The rclpy spin loop runs on the main thread and calls callbacks synchronously. Using an async httpx client inside an asyncio event loop in a separate thread caused deadlocks. A synchronous client in its own daemon thread with a blocking queue drain is simpler, more reliable, and has equivalent throughput for this use case.

**Why BackgroundTasks for RCA?**
The Claude API call takes 1-3 seconds. Making the telemetry ingest endpoint wait for it would block the bridge's flush thread and cause queue buildup. BackgroundTasks return the HTTP response immediately and run the LLM call after, keeping p99 telemetry ingest latency under 10ms.

**Why a replay buffer on the WebSocket manager?**
Without it, a browser refresh or new dashboard tab would show empty panels until the next GPU poll (3s) and the next telemetry event. The 50-event buffer means the dashboard is instantly populated with recent fleet state on any new connection.

**Why vanilla JS for the dashboard?**
Zero build tooling means the dashboard can be served as a static file from FastAPI with no npm, no webpack, no node_modules. Any engineer can open the HTML file, read it, and modify it. It also means the dashboard works immediately in any environment — WSL, remote server, Jetson — without installing frontend tooling.

---

*RoboFleet Nexus v0.1.0 · May 2026 · github.com/amitb-gpu/robofleet-nexus*
