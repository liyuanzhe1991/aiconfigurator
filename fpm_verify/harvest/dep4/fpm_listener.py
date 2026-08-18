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
for p in range(20380, 20388):
    sock.connect(f"tcp://127.0.0.1:{p}")
sock.setsockopt(zmq.SUBSCRIBE, b"")
print("fpm_listener: connected to tcp://127.0.0.1:20380-20387", flush=True)
while True:
    # 发布器 send_multipart((topic, seq, payload)):按 multipart 收帧,
    # 只解负载帧;循环体全异常免疫——监听器死亡=全链静默失明(r14 实测)
    try:
        frames = sock.recv_multipart()
        metrics = decode(frames[-1])
        if metrics is None:
            continue
        out.write(json.dumps(msgspec.to_builtins(metrics)) + "\n")
    except Exception:
        continue
