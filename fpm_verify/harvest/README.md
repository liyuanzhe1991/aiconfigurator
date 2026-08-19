# fpm-verify 真值采收(固定方法论)

r15 实战工装收编。每拓扑一个自足 kit(tep4/dep4 逐字节保真,tp4 为 tep4
去 EP 派生),统一编排,不再按战役现攒脚本。

## 流程(stage_and_run.sh <topo> <ctx>)

1. 建 Guaranteed 4 卡 pod(`<topo>/k8s_deploy.yaml`;/results 挂共享 PVC
   `_yuanli_l3_results/r16_<topo>`,pod 被抢占数据不丢);
2. kit 布置:exec-cat + 双端 sha256,5 次重试(禁 kubectl cp);
3. 单 nohup 会话串行(守护进程必须活在链内):
   `serve_stack.sh`(etcd/nats/frontend/ZMQ listener + prefill 引擎,等就绪)
   → `phase_decode.sh`(切 decode-parity 引擎,按 decode_plan 逐窗锁步收割)
   → `phase_prefill_entry.sh`(切 prefill-parity 引擎,burst + mixed)
   → `/results/phase_all.log` 落 HARVEST-ALL-DONE。

## 取件(fetch_results.sh <topo> <ctx> <outdir>)

16MB 分块 + 逐块 sha256 + 整文件 sha 复核(teleport 会话级流劣化记过档:
单流大件必截断;>4h 战役在取件前建议重登 tsh)。

## 口径不变量(勿改)

- 真值内容 = ShareGPT 奇数池;serving = parity 配置 + 原版调度器路径
  (镜像烘焙的 instrumented scheduler 在 serve 模式只透传 FPM 遥测);
- decode serve 显式 `--no-enable-prefix-caching`:防 bench 随机请求前缀
  命中使 KV 池物理缩水(与采集侧的 kvwarm/prefix 制度无冲突——r15 以
  1.70%/1.61% 实证两侧稳态语义等价);
- dep4:L3_DP_MODE=1(DP 拦路石 + 镜像门);tp4 不启用。

## 故障恢复(手册)

- 引擎死/需重启:必须全家杀 `dynamo.vllm`、**`VLLM::` worker**、
  `dynamo.frontend`、`fpm_listener`、`etcd --data-dir`、`nats-server`,
  然后轮询 nvidia-smi 至单卡显存 <1GB 才可重launch(漏杀 VLLM:: worker
  会残留 130+GiB 显存,新引擎 init_device 直接 ValueError);
- pod 被抢占:结果在 PVC,换 pod 续跑或直接取件;
- 相位标记:serve_stack.log(ENGINE-READY/DIED)、phase_decode.log
  (DECODE-PHASE-DONE)、phase_prefill.log(PREFILL-PHASE-DONE)、
  phase_all.log(HARVEST-ALL-DONE)。

## 后续打分

../scoring/score_decode.py 与 score_prefill_burst.py(参数化模型/系统/
backend),判据见 ../../fpm_e2e_20260811/R16_ACCEPTANCE_SPEC.md §4。

## 同机协议(2026-08-20 拍板,必须遵守)

真值收割必须与被验采集**同一台物理机**:健康节点间小坐标测速离散 4-6%
(时钟守卫盲区),跨机打分带 ±3-4% 地板,只作参考不作判定。操作:
采集 cell 开跑时记下其 pod 的 spec.nodeName(aic 不记,自己记);
收割用 `PIN_NODE=<node> bash stage_and_run.sh ...` 钉死同节点。
stage_and_run 会把实际节点写进 /results/nodeName.txt 并打印 HARVEST-NODE。
未设 PIN_NODE 会打警告并落随机节点。
