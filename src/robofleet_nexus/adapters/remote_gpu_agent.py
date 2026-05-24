"""
Remote GPU agent for RoboFleet Nexus.

Run this on your RTX-5080 laptop (or any remote GPU node) to push
that machine's GPU metrics to a central Nexus instance running elsewhere.

USAGE:
    # On the RTX-5080 WSL machine:
    python -m robofleet_nexus.adapters.remote_gpu_agent \
        --nexus-url http://<A1000-IP>:8000 \
        --node-id rtx5080-node \
        --interval 3

The GPU data will appear in the Nexus dashboard GPU inventory panel
alongside the local GPUs on the Nexus host.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket

import httpx

from robofleet_nexus.core.gpu_monitor import query_gpus_async

logger = logging.getLogger("robofleet.remote_gpu_agent")


async def _push_loop(
    nexus_url: str,
    node_id: str,
    interval: float,
) -> None:
    hostname = socket.gethostname()
    logger.info(
        "Remote GPU agent: node=%s  host=%s  target=%s  interval=%.1fs",
        node_id, hostname, nexus_url, interval,
    )

    async with httpx.AsyncClient(base_url=nexus_url, timeout=5.0) as client:
        while True:
            devices = await query_gpus_async()
            if devices:
                payload = {
                    "node_id": node_id,
                    "hostname": hostname,
                    "gpus": [d.to_dict() for d in devices],
                }
                try:
                    r = await client.post(
                        "/gpu/remote-snapshot",
                        content=json.dumps(payload),
                        headers={"Content-Type": "application/json"},
                    )
                    if r.status_code not in (200, 201, 204):
                        logger.warning("Push %s: %s", r.status_code, r.text[:80])
                    else:
                        logger.debug(
                            "Pushed %d GPU(s) from %s", len(devices), node_id
                        )
                except httpx.ConnectError:
                    logger.warning("Cannot reach Nexus at %s", nexus_url)
                except Exception as exc:
                    logger.debug("Push error: %s", exc)
            else:
                logger.warning("No GPUs detected on this node — is nvidia-smi available?")

            await asyncio.sleep(interval)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )
    parser = argparse.ArgumentParser(description="RoboFleet Nexus remote GPU agent")
    parser.add_argument("--nexus-url", required=True, help="URL of the central Nexus server")
    parser.add_argument("--node-id", default=socket.gethostname(), help="Identifier for this GPU node")
    parser.add_argument("--interval", type=float, default=3.0, help="Poll interval in seconds")
    args = parser.parse_args()

    try:
        asyncio.run(_push_loop(args.nexus_url, args.node_id, args.interval))
    except KeyboardInterrupt:
        logger.info("Remote GPU agent stopped")


if __name__ == "__main__":
    main()
