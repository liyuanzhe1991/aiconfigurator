# Capture the FPM stream (real-traffic per-step ForwardPassMetrics) to JSONL.
# Uses the image's own decoder — zero format assumptions.
import json
import sys

import msgspec
import zmq

from dynamo.common.forward_pass_metrics import decode

out = open(sys.argv[1], "a", buffering=1)
ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://127.0.0.1:20380")
sock.setsockopt(zmq.SUBSCRIBE, b"")
print("fpm_listener: connected to tcp://127.0.0.1:20380", flush=True)
while True:
    data = sock.recv()
    metrics = decode(data)
    if metrics is None:
        continue
    out.write(json.dumps(msgspec.to_builtins(metrics)) + "\n")
