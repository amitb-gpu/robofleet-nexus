# Terminal 1 — API server
cd ~/robofleet-nexus
conda activate robofleet-nexus
export $(grep -v '^#' ~/.env | grep ANTHROPIC_API_KEY | tr -d ' ')
uvicorn robofleet_nexus.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — mock bridge (or real ROS2 bridge)
cd ~/robofleet-nexus
conda activate robofleet-nexus
python -m robofleet_nexus.adapters.ros2_bridge --mock --robot-id bot_001

# Browser
http://localhost:8000/dashboard

# To trigger RCA immediately without waiting for battery to drain:
curl -s -X POST http://localhost:8000/telemetry \
  -H "Content-Type: application/json" \
  -d '{"event_id":"demo-001","robot_id":"bot_001","source":"mock","event_type":"battery_state","subsystem":"power","message":"Battery at 14.0%","metrics":{"battery_pct":14.0},"metadata":{}}'
