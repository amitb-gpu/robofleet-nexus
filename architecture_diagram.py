"""
RoboFleet Nexus — Architecture Diagram
Run:  python3 robofleet_diagram.py
Output: robofleet_nexus_architecture.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = "#0D0F14"
PANEL_DARK  = "#13161D"
PANEL_MID   = "#1A1E28"

BLUE        = "#2E6FD8"
BLUE_LIGHT  = "#4F9CF9"
BLUE_DARK   = "#1B3A6B"
GREEN       = "#34D399"
AMBER       = "#FBBF24"
RED         = "#F87171"
PURPLE      = "#A78BFA"
CYAN        = "#22D3EE"
PINK        = "#F472B6"

TEXT_HI     = "#E8EAF0"
TEXT_MID    = "#9CA3AF"
TEXT_LO     = "#6B7280"

LAYER_ALPHA = 0.18

fig, ax = plt.subplots(figsize=(20, 13))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis("off")

# ── Helpers ───────────────────────────────────────────────────────────────────

def box(x, y, w, h, color, alpha=1.0, radius=0.25, lw=1.2, ls="-"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, linestyle=ls,
        edgecolor=color, facecolor=(*matplotlib.colors.to_rgb(color), alpha * 0.18),
        zorder=3,
    )
    ax.add_patch(rect)

def solid_box(x, y, w, h, fill, edge, radius=0.2, lw=1.0, ls="-"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, linestyle=ls,
        edgecolor=edge, facecolor=fill,
        zorder=4,
    )
    ax.add_patch(rect)

def label(x, y, text, color=TEXT_HI, size=8.5, bold=False, ha="center", va="center", zorder=5):
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, color=color, fontsize=size, fontweight=weight,
            ha=ha, va=va, zorder=zorder, fontfamily="monospace")

def section_label(x, y, text, color):
    ax.text(x, y, text.upper(), color=color, fontsize=7, fontweight="bold",
            ha="left", va="center", zorder=5, alpha=0.7,
            fontfamily="monospace", letter_spacing=1)

def arrow(x1, y1, x2, y2, color, lw=1.5, style="->", alpha=0.7):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color, lw=lw, alpha=alpha,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=6,
    )

def curved_arrow(x1, y1, x2, y2, color, rad=0.2, lw=1.5, alpha=0.7):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            color=color, lw=lw, alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=6,
    )

def divider(y, color=PANEL_MID, alpha=0.6):
    ax.axhline(y, color=color, lw=0.5, alpha=alpha, zorder=2)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(10, 12.55, "RoboFleet Nexus", color=TEXT_HI, fontsize=22, fontweight="bold",
        ha="center", va="center", fontfamily="monospace")
ax.text(10, 12.15, "Platform Architecture  ·  v0.1.0  ·  ROS2 Jazzy · FastAPI · Claude API · NVIDIA GPU",
        color=TEXT_MID, fontsize=8, ha="center", va="center", fontfamily="monospace")

# ── Layer backgrounds ─────────────────────────────────────────────────────────
# Layer 1 — Hardware
solid_box(0.3, 0.3, 19.4, 1.55, (*matplotlib.colors.to_rgb(AMBER), 0.06), AMBER, radius=0.3, lw=0.7)
ax.text(0.65, 1.62, "HARDWARE LAYER", color=AMBER, fontsize=6.5, fontweight="bold",
        alpha=0.8, fontfamily="monospace")

# Layer 2 — ROS2 / Ingestion
solid_box(0.3, 2.05, 19.4, 2.1, (*matplotlib.colors.to_rgb(CYAN), 0.05), CYAN, radius=0.3, lw=0.7)
ax.text(0.65, 3.85, "INGESTION LAYER", color=CYAN, fontsize=6.5, fontweight="bold",
        alpha=0.8, fontfamily="monospace")

# Layer 3 — API Core
solid_box(0.3, 4.35, 19.4, 2.8, (*matplotlib.colors.to_rgb(BLUE_LIGHT), 0.06), BLUE_LIGHT, radius=0.3, lw=0.7)
ax.text(0.65, 6.85, "API CORE LAYER", color=BLUE_LIGHT, fontsize=6.5, fontweight="bold",
        alpha=0.8, fontfamily="monospace")

# Layer 4 — AI / Intelligence
solid_box(0.3, 7.35, 19.4, 2.0, (*matplotlib.colors.to_rgb(PURPLE), 0.06), PURPLE, radius=0.3, lw=0.7)
ax.text(0.65, 9.05, "INTELLIGENCE LAYER", color=PURPLE, fontsize=6.5, fontweight="bold",
        alpha=0.8, fontfamily="monospace")

# Layer 5 — Presentation
solid_box(0.3, 9.55, 19.4, 2.25, (*matplotlib.colors.to_rgb(GREEN), 0.06), GREEN, radius=0.3, lw=0.7)
ax.text(0.65, 11.5, "PRESENTATION LAYER", color=GREEN, fontsize=6.5, fontweight="bold",
        alpha=0.8, fontfamily="monospace")

# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

# RTX A1000
solid_box(0.7, 0.5, 3.2, 1.15, "#1a1410", AMBER, radius=0.18, lw=1.2)
label(2.3, 1.3, "RTX A1000 6GB", AMBER, 8.5, bold=True)
label(2.3, 1.02, "Laptop GPU · WSL2", TEXT_MID, 7.5)
label(2.3, 0.76, "WORKSTATION_DEV", TEXT_LO, 7)

# RTX 5080
solid_box(4.2, 0.5, 3.2, 1.15, "#1a1410", AMBER, radius=0.18, lw=1.2)
label(5.8, 1.3, "RTX 5080 16GB", AMBER, 8.5, bold=True)
label(5.8, 1.02, "Laptop GPU · WSL2", TEXT_MID, 7.5)
label(5.8, 0.76, "WORKSTATION_HIGH", TEXT_LO, 7)

# Robot physical
solid_box(7.7, 0.5, 3.0, 1.15, "#101a14", GREEN, radius=0.18, lw=1.2)
label(9.2, 1.3, "Physical Robot", GREEN, 8.5, bold=True)
label(9.2, 1.02, "ROS2 Jazzy · Ubuntu 24", TEXT_MID, 7.5)
label(9.2, 0.76, "/battery /joints /odom", TEXT_LO, 7)

# Gazebo / Isaac Sim
solid_box(11.0, 0.5, 3.0, 1.15, "#101018", PURPLE, radius=0.18, lw=1.2)
label(12.5, 1.3, "Isaac Sim / Gazebo", PURPLE, 8.5, bold=True)
label(12.5, 1.02, "GPU-accelerated sim", TEXT_MID, 7.5)
label(12.5, 0.76, "IsaacSimulationJob", TEXT_LO, 7)

# Claude API (external)
solid_box(14.3, 0.5, 3.0, 1.15, "#180f1a", PINK, radius=0.18, lw=1.2)
label(15.8, 1.3, "Claude API", PINK, 8.5, bold=True)
label(15.8, 1.02, "claude-sonnet-4", TEXT_MID, 7.5)
label(15.8, 0.76, "console.anthropic.com", TEXT_LO, 7)

# nvidia-smi
solid_box(17.6, 0.5, 1.9, 1.15, "#1a1810", AMBER, radius=0.18, lw=1.2)
label(18.55, 1.3, "nvidia-smi", AMBER, 8, bold=True)
label(18.55, 1.02, "GPU metrics", TEXT_MID, 7)
label(18.55, 0.76, "poll 3s", TEXT_LO, 7)

# ═══════════════════════════════════════════════════════════════════════════════
# INGESTION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

# ROS2 Bridge
solid_box(0.7, 2.2, 4.0, 1.6, "#0e1418", CYAN, radius=0.18, lw=1.2)
label(2.7, 3.55, "ros2_bridge.py", CYAN, 9, bold=True)
label(2.7, 3.22, "rclpy · system Python 3.12", TEXT_MID, 7.5)
label(2.7, 2.96, "subs: /battery_state", TEXT_LO, 7)
label(2.7, 2.72, "      /diagnostics", TEXT_LO, 7)
label(2.7, 2.48, "      /joint_states  /odom", TEXT_LO, 7)

# Mock Bridge
solid_box(5.0, 2.2, 3.0, 1.6, "#0e1418", BLUE_LIGHT, radius=0.18, lw=1.2)
label(6.5, 3.55, "mock_bridge.py", BLUE_LIGHT, 9, bold=True)
label(6.5, 3.22, "Dev / CI mode", TEXT_MID, 7.5)
label(6.5, 2.96, "Synthetic telemetry", TEXT_LO, 7)
label(6.5, 2.72, "Battery drain + joints", TEXT_LO, 7)
label(6.5, 2.48, "Circle odometry path", TEXT_LO, 7)

# Remote GPU Agent
solid_box(8.3, 2.2, 3.2, 1.6, "#0e1418", AMBER, radius=0.18, lw=1.2)
label(9.9, 3.55, "remote_gpu_agent.py", AMBER, 9, bold=True)
label(9.9, 3.22, "RTX 5080 node", TEXT_MID, 7.5)
label(9.9, 2.96, "nvidia-smi remote poll", TEXT_LO, 7)
label(9.9, 2.72, "Push to central Nexus", TEXT_LO, 7)
label(9.9, 2.48, "interval: 3s", TEXT_LO, 7)

# Isaac Job Planner (ingestion side)
solid_box(11.8, 2.2, 3.2, 1.6, "#100e18", PURPLE, radius=0.18, lw=1.2)
label(13.4, 3.55, "isaac/scheduler.py", PURPLE, 9, bold=True)
label(13.4, 3.22, "GPU-aware job planner", TEXT_MID, 7.5)
label(13.4, 2.96, "VRAM req. matching", TEXT_LO, 7)
label(13.4, 2.72, "SimulationProfile", TEXT_LO, 7)
label(13.4, 2.48, "POST /simulations/plan", TEXT_LO, 7)

# GPU Monitor
solid_box(15.3, 2.2, 2.8, 1.6, "#141208", AMBER, radius=0.18, lw=1.2)
label(16.7, 3.55, "gpu_monitor.py", AMBER, 9, bold=True)
label(16.7, 3.22, "Background task", TEXT_MID, 7.5)
label(16.7, 2.96, "GpuDevice + profile", TEXT_LO, 7)
label(16.7, 2.72, "classify() by VRAM", TEXT_LO, 7)
label(16.7, 2.48, "asyncio poll loop", TEXT_LO, 7)

# Audit Log
solid_box(18.35, 2.2, 1.6, 1.6, "#0e1410", GREEN, radius=0.18, lw=1.2)
label(19.15, 3.55, "audit/", GREEN, 8.5, bold=True)
label(19.15, 3.22, "hash_chain", GREEN, 8, bold=True)
label(19.15, 2.96, "SHA-256", TEXT_LO, 7)
label(19.15, 2.72, "tamper-", TEXT_LO, 7)
label(19.15, 2.48, "evident", TEXT_LO, 7)

# ═══════════════════════════════════════════════════════════════════════════════
# API CORE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

# FastAPI main
solid_box(0.7, 4.5, 5.5, 2.4, "#0c1020", BLUE_LIGHT, radius=0.22, lw=1.5)
label(3.45, 6.55, "FastAPI  ·  main.py", BLUE_LIGHT, 10, bold=True)
label(3.45, 6.22, "uvicorn · host 0.0.0.0:8000", TEXT_MID, 8)

label(1.5,  5.85, "POST /telemetry", BLUE_LIGHT, 7.5, ha="left")
label(1.5,  5.62, "GET  /diagnostics", TEXT_MID,  7.5, ha="left")
label(1.5,  5.39, "GET  /audit", TEXT_MID,  7.5, ha="left")
label(1.5,  5.16, "GET  /gpu/inventory", TEXT_MID,  7.5, ha="left")
label(1.5,  4.93, "POST /simulations/plan", TEXT_MID, 7.5, ha="left")
label(1.5,  4.70, "POST /rca/analyze", PURPLE, 7.5, ha="left")
label(1.5,  4.47, "GET  /rca", PURPLE,  7.5, ha="left")

# Pydantic schemas
solid_box(6.5, 4.5, 3.5, 2.4, "#0e1018", BLUE, radius=0.18, lw=1.2)
label(8.25, 6.55, "telemetry/schemas.py", BLUE, 9, bold=True)
label(8.25, 6.22, "Pydantic v2 models", TEXT_MID, 7.5)
label(8.25, 5.96, "RobotEvent", TEXT_LO, 7.5)
label(8.25, 5.72, "DiagnosticFinding", TEXT_LO, 7.5)
label(8.25, 5.48, "RobotSeverity", TEXT_LO, 7.5)
label(8.25, 5.24, "GpuInventory", TEXT_LO, 7.5)
label(8.25, 5.00, "SimulationPlan", TEXT_LO, 7.5)
label(8.25, 4.76, "IsaacSimulationJob", TEXT_LO, 7.5)

# Diagnostic Rules
solid_box(10.3, 4.5, 3.5, 2.4, "#100e10", RED, radius=0.18, lw=1.2)
label(12.05, 6.55, "diagnostics/rules.py", RED, 9, bold=True)
label(12.05, 6.22, "evaluate_event()", TEXT_MID, 7.5)
label(12.05, 5.96, "battery_pct ≤ 15% → critical", TEXT_LO, 7)
label(12.05, 5.72, "battery_pct ≤ 25% → warning", TEXT_LO, 7)
label(12.05, 5.48, "motor_temp ≥ 90°C → critical", TEXT_LO, 7)
label(12.05, 5.24, "gpu_temp ≥ 85°C → warning", TEXT_LO, 7)
label(12.05, 5.00, "packet_loss ≥ 5% → warning", TEXT_LO, 7)
label(12.05, 4.76, "joint.eff ≥ 40Nm → warning", TEXT_LO, 7)

# WebSocket Manager
solid_box(14.1, 4.5, 3.0, 2.4, "#0c1210", GREEN, radius=0.18, lw=1.2)
label(15.6, 6.55, "ws_manager.py", GREEN, 9, bold=True)
label(15.6, 6.22, "ConnectionManager", TEXT_MID, 7.5)
label(15.6, 5.96, "50-event replay buf", TEXT_LO, 7)
label(15.6, 5.72, "emit_telemetry()", TEXT_LO, 7)
label(15.6, 5.48, "emit_gpu_snapshot()", TEXT_LO, 7)
label(15.6, 5.24, "emit_diagnostic()", TEXT_LO, 7)
label(15.6, 5.00, "emit_rca_result()", TEXT_LO, 7)
label(15.6, 4.76, "broadcast() → all WS", TEXT_LO, 7)

# WS Routes
solid_box(17.4, 4.5, 2.2, 2.4, "#0c1210", CYAN, radius=0.18, lw=1.2)
label(18.5, 6.55, "ws_routes.py", CYAN, 9, bold=True)
label(18.5, 6.22, "FastAPI lifespan", TEXT_MID, 7.5)
label(18.5, 5.96, "GET /dashboard", TEXT_LO, 7)
label(18.5, 5.72, "WS  /ws", TEXT_LO, 7)
label(18.5, 5.48, "gpu_poll_loop()", TEXT_LO, 7)
label(18.5, 5.24, "on_telemetry()", TEXT_LO, 7)
label(18.5, 5.00, "on_diagnostic()", TEXT_LO, 7)
label(18.5, 4.76, "on_rca_result()", TEXT_LO, 7)

# ═══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

# RCA Agent
solid_box(0.7, 7.5, 8.5, 1.65, "#100a18", PURPLE, radius=0.22, lw=1.5)
label(4.95, 8.9, "rca/agent.py  —  AI Root Cause Analysis Engine", PURPLE, 10, bold=True)
label(4.95, 8.58, "AsyncAnthropic client  ·  claude-sonnet-4  ·  structured JSON output", TEXT_MID, 8)
label(1.5,  8.22, "Input:", TEXT_LO, 7.5, ha="left")
label(2.3,  8.22, "list[DiagnosticFinding]  +  last 10 RobotEvents", TEXT_HI, 7.5, ha="left")
label(1.5,  7.97, "Output:", TEXT_LO, 7.5, ha="left")
label(2.3,  7.97, "summary · root_causes (ranked, confidence) · risk_level", TEXT_HI, 7.5, ha="left")
label(1.5,  7.72, "", TEXT_LO, 7.5, ha="left")
label(2.3,  7.72, "recommended_actions (immediate/soon/monitor) · ETA · tokens", TEXT_HI, 7.5, ha="left")

# System prompt highlight
solid_box(9.5, 7.5, 4.5, 1.65, "#0c0c14", BLUE, radius=0.18, lw=1.0)
label(11.75, 8.9, "System Prompt Context", BLUE, 9, bold=True)
label(11.75, 8.58, "Robotics diagnostics engineer persona", TEXT_MID, 7.5)
label(11.75, 8.3, "JSON-only response schema", TEXT_LO, 7)
label(11.75, 8.06, "Safety-first guidance", TEXT_LO, 7)
label(11.75, 7.82, "Hardware damage / injury flags", TEXT_LO, 7)
label(11.75, 7.58, "Token usage tracking", TEXT_LO, 7)

# RCA results store
solid_box(14.3, 7.5, 2.8, 1.65, "#0e100e", GREEN, radius=0.18, lw=1.0)
label(15.7, 8.9, "RCA Results", GREEN, 9, bold=True)
label(15.7, 8.58, "In-memory store", TEXT_MID, 7.5)
label(15.7, 8.3, "GET /rca", TEXT_LO, 7)
label(15.7, 8.06, "list[dict]", TEXT_LO, 7)
label(15.7, 7.82, "rca_id · robot_id", TEXT_LO, 7)
label(15.7, 7.58, "timestamp · findings", TEXT_LO, 7)

# Background task
solid_box(17.4, 7.5, 2.2, 1.65, "#0e100e", AMBER, radius=0.18, lw=1.0)
label(18.5, 8.9, "BackgroundTask", AMBER, 9, bold=True)
label(18.5, 8.58, "FastAPI async", TEXT_MID, 7.5)
label(18.5, 8.3, "Non-blocking", TEXT_LO, 7)
label(18.5, 8.06, "fires on every", TEXT_LO, 7)
label(18.5, 7.82, "DiagnosticFinding", TEXT_LO, 7)
label(18.5, 7.58, "→ run_rca()", TEXT_LO, 7)

# ═══════════════════════════════════════════════════════════════════════════════
# PRESENTATION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

# Dashboard HTML
solid_box(0.7, 9.7, 6.0, 1.9, "#0a1410", GREEN, radius=0.22, lw=1.5)
label(3.7, 11.35, "dashboard/index.html  —  Live Fleet Dashboard", GREEN, 10, bold=True)
label(3.7, 11.02, "Vanilla JS + WebSocket  ·  zero build step  ·  served at GET /dashboard", TEXT_MID, 8)

label(1.1, 10.7,  "[>] GPU inventory cards", TEXT_HI, 7.5, ha="left")
label(1.1, 10.46, "[>] Robot fleet table", TEXT_HI, 7.5, ha="left")
label(1.1, 10.22, "[>] Live event stream (200 events)", TEXT_HI, 7.5, ha="left")
label(4.5, 10.7,  "[>] Diagnostic findings panel", TEXT_HI, 7.5, ha="left")
label(4.5, 10.46, "[>] AI RCA cards (risk-coloured)", TEXT_HI, 7.5, ha="left")
label(4.5, 10.22, "[>] Simulation job queue", TEXT_HI, 7.5, ha="left")

# KPI chips
for i, (val, lbl, col) in enumerate([
    ("1", "Active robots", BLUE_LIGHT),
    ("1", "GPU devices",   GREEN),
    ("—", "Sim jobs",      AMBER),
    ("—", "Diagnostics",   PURPLE),
]):
    bx = 1.0 + i * 1.45
    solid_box(bx, 9.72, 1.3, 0.42, "#0d1218", col, radius=0.1, lw=0.8)
    label(bx + 0.65, 9.93, f"{val}  {lbl}", col, 6.5)

# Browser client
solid_box(7.1, 9.7, 2.8, 1.9, "#0a0e14", CYAN, radius=0.18, lw=1.2)
label(8.5, 11.35, "Browser Client", CYAN, 9, bold=True)
label(8.5, 11.02, "Chrome / Firefox", TEXT_MID, 7.5)
label(8.5, 10.72, "ws://localhost:8000/ws", TEXT_LO, 7)
label(8.5, 10.48, "50-event replay on", TEXT_LO, 7)
label(8.5, 10.24, "reconnect", TEXT_LO, 7)
label(8.5, 10.00, "F12 console debug", TEXT_LO, 7)

# Prometheus (future)
solid_box(10.2, 9.7, 3.0, 1.9, "#141410", AMBER, radius=0.18, lw=1.2, ls="--")
label(11.7, 11.35, "Prometheus / Grafana", AMBER, 9, bold=True)
label(11.7, 11.02, "Phase 2 · planned", TEXT_LO, 7.5)
label(11.7, 10.72, "GET /metrics", TEXT_LO, 7)
label(11.7, 10.48, "GPU gauges", TEXT_LO, 7)
label(11.7, 10.24, "telemetry counters", TEXT_LO, 7)
label(11.7, 10.00, "diagnostic rates", TEXT_LO, 7)

# Kubernetes (future)
solid_box(13.5, 9.7, 3.0, 1.9, "#0e100e", GREEN, radius=0.18, lw=1.2, ls="--")
label(15.0, 11.35, "Kubernetes Operator", GREEN, 9, bold=True)
label(15.0, 11.02, "Phase 3 · planned", TEXT_LO, 7.5)
label(15.0, 10.72, "SimulationJob CRD", TEXT_LO, 7)
label(15.0, 10.48, "RobotFleet CRD", TEXT_LO, 7)
label(15.0, 10.24, "GPU node provisioning", TEXT_LO, 7)
label(15.0, 10.00, "Helm chart", TEXT_LO, 7)

# Quantum planner (future)
solid_box(16.8, 9.7, 2.8, 1.9, "#14100e", PINK, radius=0.18, lw=1.2, ls="--")
label(18.2, 11.35, "Quantum Router", PINK, 9, bold=True)
label(18.2, 11.02, "Phase 3 · planned", TEXT_LO, 7.5)
label(18.2, 10.72, "QUBO encoding", TEXT_LO, 7)
label(18.2, 10.48, "QAOA via Cirq+TFQ", TEXT_LO, 7)
label(18.2, 10.24, "cuQuantum / RTX", TEXT_LO, 7)
label(18.2, 10.00, "multi-robot paths", TEXT_LO, 7)

# ═══════════════════════════════════════════════════════════════════════════════
# ARROWS
# ═══════════════════════════════════════════════════════════════════════════════

# Hardware → Ingestion
arrow(2.3, 1.65, 2.3, 2.2,   CYAN,   lw=1.5)   # Robot → ROS2 bridge
arrow(9.2, 1.65, 9.5, 2.2,   CYAN,   lw=1.5)   # Robot → ROS2 bridge (phys)
arrow(5.8, 1.65, 9.7, 2.2,   AMBER,  lw=1.2)   # RTX5080 → remote agent
arrow(18.55,1.65,16.9,2.2,   AMBER,  lw=1.2)   # nvidia-smi → GPU monitor
arrow(12.5, 1.65, 13.3, 2.2, PURPLE, lw=1.2)   # Isaac → scheduler
arrow(15.8, 1.65, 15.8, 2.2, PINK,   lw=1.2)   # Claude → (future)

# Ingestion → API Core
arrow(2.7,  3.8,  2.7,  4.5,  CYAN,       lw=1.5)  # ROS2 bridge → FastAPI
arrow(6.5,  3.8,  3.5,  4.5,  BLUE_LIGHT, lw=1.2)  # Mock → FastAPI
arrow(9.9,  3.8,  9.9,  4.5,  AMBER,      lw=1.2)  # GPU agent → ws_manager
arrow(13.4, 3.8,  13.4, 4.5,  PURPLE,     lw=1.2)  # scheduler → FastAPI
arrow(16.7, 3.8,  16.7, 4.5,  AMBER,      lw=1.2)  # GPU monitor → ws_routes
arrow(19.15,3.8,  19.0, 4.5,  GREEN,      lw=1.2)  # audit → ws_routes

# FastAPI → downstream
arrow(5.5,  5.7,  6.5,  5.7,  BLUE,   lw=1.2)   # FastAPI → schemas
arrow(8.0,  5.7, 10.3,  5.7,  RED,    lw=1.2)   # schemas → rules
arrow(3.45, 4.5,  3.45, 7.5,  PURPLE, lw=1.5)   # FastAPI → RCA (vertical)
arrow(13.8, 5.7, 14.1,  5.7,  GREEN,  lw=1.2)   # rules → ws_manager
arrow(17.1, 5.7, 17.4,  5.7,  CYAN,   lw=1.2)   # ws_mgr → ws_routes

# API Core → Intelligence
arrow(4.95, 7.5, 4.95, 7.15, PURPLE, lw=1.5)   # RCA agent ← FastAPI bg task
arrow(9.2,  7.5, 9.5,  7.5,  BLUE,   lw=1.2)   # agent → system prompt
arrow(14.1, 7.5, 14.3, 7.5,  GREEN,  lw=1.2)   # agent → results store
arrow(17.2, 7.5, 17.4, 7.5,  AMBER,  lw=1.2)   # agent → bg task

# RCA → Claude API (hardware layer)
curved_arrow(4.95, 7.5, 15.8, 1.65, PINK, rad=-0.15, lw=1.5, alpha=0.6)

# Intelligence → Presentation
arrow(5.5,  7.5,  5.5,  9.7,  GREEN,  lw=1.5)   # RCA → dashboard
arrow(15.7, 9.15, 15.7, 9.55, AMBER,  lw=1.0)   # results → (dashed future)

# WS Routes → Dashboard
arrow(18.5, 6.9,  8.5,  9.7,  GREEN,  lw=1.5)   # ws_routes → browser

# GPU monitor → Dashboard (via ws)
curved_arrow(16.7, 4.5, 8.2, 9.7, AMBER, rad=0.1, lw=1.2, alpha=0.5)

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (CYAN,       "ROS2 / Real-time data"),
    (BLUE_LIGHT, "FastAPI endpoints"),
    (RED,        "Diagnostic rules"),
    (PURPLE,     "AI / RCA"),
    (GREEN,      "WebSocket / Dashboard"),
    (AMBER,      "GPU / Hardware"),
    (PINK,       "External API"),
    (TEXT_LO,    "Phase 2/3 (planned)"),
]
ax.text(0.5, 0.18, "Legend:", color=TEXT_MID, fontsize=7, fontfamily="monospace", va="center")
for i, (col, lbl) in enumerate(legend_items):
    x = 1.4 + i * 2.32
    solid_box(x, 0.08, 0.28, 0.22, (*matplotlib.colors.to_rgb(col), 0.25), col, radius=0.05, lw=0.8)
    ax.text(x + 0.38, 0.19, lbl, color=TEXT_MID, fontsize=6.5, va="center", fontfamily="monospace")

plt.tight_layout(pad=0.2)
plt.savefig("robofleet_nexus_architecture.png", dpi=180, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print("Saved: robofleet_nexus_architecture.png")
