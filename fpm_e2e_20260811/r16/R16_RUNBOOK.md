# R16 复现手册 — H200×MiniMax-M2.7 四卡全并行,纯 AIC 官方命令(2026-08-18 实测)

任何人按本文逐字执行可复现"采集→建库→预测"。对齐(C4)属独立 fpm_verify
套件,另见 ../../fpm_verify/VERIFY_RUNBOOK.md。

## 1. 版本与镜像(全部钉死)

| 项 | 值 |
|---|---|
| 代码树 | r16-acceptance @ **68707ffb**(= 最新 main ⊕ #1473 delta @82d83e50 ⊕ #1475 @dfc7896d ⊕ 三修复 8b5d4399/8552f522/68707ffb) |
| 引擎镜像 | `nvcr.io/0980761089281446/dynamo-fpm-frozen:gc-timing-20260818` |
| 镜像 digest | `sha256:adcd28a943c7811e71931a4c154d31cb95d126469bc9b4fe2a8981f5327e8c48` |
| 镜像内容 | kvwarm 11 手术 + 真实内容池 v4(ShareGPT 偶数池)+ argsdump + timing.phases 相位仪表(零测量开销,5-boot 验证) |
| 模型 | MiniMaxAI/MiniMax-M2.7(fp8_block,227.69B,snapshot d494266a4affc0d2995ba1fa35c8481cbd84294b) |
| 集群 | nebius-2(H200 ×8/节点),namespace yuanli-aic,PVC model-cache |
| backend | vllm,库版本 0.25.1 |

## 2. 环境(一次)

```bash
tsh login --proxy=nv-prd-dgxc.teleport.sh
PY=<venv>/bin/python                       # 含 collector 依赖
export PYTHONPATH=$PWD/src:$PWD/aic-core/src   # 仓库根
# native 核须与代码同版本:cd aic-core && uvx maturin build --release
# (cargo 在 /opt/homebrew/opt/rustup/bin;wheel 解出 _aiconfigurator_core.abi3.so
#  放 aic-core/src/aiconfigurator_core/)
export FPM_KUBECTL="kubectl --context=nv-prd-dgxc.teleport.sh-dynamo-nebius-2"
```

## 3. C1 采集命令(字面,三形;先冒烟后正式)

公共尾部 `<TAIL>`:
```
--model-path MiniMaxAI/MiniMax-M2.7 --gpu h200_sxm --namespace yuanli-aic \
--model-cache model-cache:/workspace/model_cache:models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a4affc0d2995ba1fa35c8481cbd84294b \
--image-pull-secret nvcr-push-secret \
--generator-set K8sConfig.k8s_image=nvcr.io/0980761089281446/dynamo-fpm-frozen:gc-timing-20260818 \
--generator-set 'K8sConfig.extra_env=[{"name":"DYN_BENCH_PREFILL_CONTENT","value":"sharegpt"}]' \
--generator-set 'K8sConfig.fpm_resource_labels={"kai.scheduler/queue":"dynamo"}' \
--generator-set 'K8sConfig.worker_extra_pod_spec={"schedulerName":"kai-scheduler","securityContext":{"runAsUser":0,"runAsGroup":0}}' \
--fpm-database-root "$PWD/fpm_formal_database"
```

```bash
# 冒烟(不写库;--limit 2 覆盖 prefill+decode 两 cell)
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --fpm-max-gpus 4 --fpm-parallel-presets tep --fpm-tp-sizes 4 <TAIL 去 database-root> --smoke --limit 2

# 正式三连(串行;复跑合并同一 parquet,sha 封印 + first-publisher-wins)
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --fpm-max-gpus 4 --fpm-parallel-presets tep     --fpm-tp-sizes 4 <TAIL>   # tep4: tp4/ep4
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --fpm-max-gpus 4 --fpm-parallel-presets dep     --fpm-dp-sizes 4 <TAIL>   # dep4: dp4/ep4
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --fpm-max-gpus 4 --fpm-parallel-presets pure_tp --fpm-tp-sizes 4 <TAIL>   # tp4: tp4/moe_tp4
```

**输入**:模型 snapshot(PVC)、ShareGPT 偶数池(镜像内容源,
DYN_BENCH_PREFILL_CONTENT=sharegpt)、引擎网格(产品自动生成,边界封顶)。
**输出**(每 run):
- `fpm_formal_database/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet`(+ .metadata.json 封印)
- `fpm_forward_artifacts/<run>/run-manifest.json`(schema aic_fpm_run_manifest v1)
- `fpm_forward_artifacts/<run>/cells/<cell>/raw/<pod>/benchmark.json`(timing.phases/kvwarm.stages)
- `resolved-config-node0.json`、渲染的 `k8s_deploy.yaml`/`run.sh`

## 4. 实测耗时(2026-08-18,守恒差 0.0%)

| Topology | Cell | Cell total | Engine startup & teardown | KV-cache warm-up | Measured inference | Input gen & KV seeding | Benchmark protocol overhead | Scheduling & artifact retrieval |
|---|---|---|---|---|---|---|---|---|
| tep4 | prefill | 24.9 min | 6.2 | — | 9.9 | 3.2 | 3.9 | 1.7 |
| tep4 | decode | 88.6 min | 4.4 | **72.2 (81%)** | 1.0 | 0.2 | 8.9 | 2.0 |
| dep4 | prefill | 37.5 min | 7.2 | — | 21.3 | 2.7 | 4.3 | 2.1 |
| dep4 | decode | 53.8 min | 5.6 | **37.4 (70%)** | 1.1 | 0.2 | 7.6 | 1.9 |
| tp4 | prefill | 24.1 min | 5.4 | — | 9.8 | 3.3 | 3.9 | 1.7 |
| tp4 | decode | 11.4 min | 4.3 | — | 0.8 | 0.2 | 4.5 | 1.6 |
| **Total** | | **240.2 min (4.0 h)** | 33.0 | 109.6 (46%) | 43.8 | 12.6 | 33.1 | 11.0 |

Column semantics: *Engine startup & teardown* = process launch → benchmark
start, plus shutdown (weight load, init, cudagraph capture); *KV-cache
warm-up* = real-KV chain building before decode measurement; *Measured
inference* = wall-clock sum of measured forward steps (the measurement
itself); *Input gen & KV seeding* = ShareGPT input-window generation +
prefix seeding for kv>0 prefill points; *Benchmark protocol overhead* =
non-measured protocol steps (request-admission beats, fake block-table
fabrication, extra median beats, point transitions, bookkeeping);
*Scheduling & artifact retrieval* = pod scheduling, staging, artifact
fetch.

数据源:run-manifest(cell_total/collector 相位/engine_timing)+ benchmark JSON
(timing.phases、kvwarm.stages[].build_seconds、measured_iteration_seconds)。

## 5. C2 建库(发布内建)

无独立命令:每次 collect 成功即聚合发布。三 run 合并结果:**32,797 行**
(tep4 9505p+1557d;dep4 9308p+1365d;tp4 9505p+1557d)。坐标合并语义:
钳位重复按物理坐标一坐标一行(tep4 1659 样本→1557 行,102 巨点合并)。

## 6. C3 预测命令(字面)

库指向:`--systems-paths <root>`,root 结构 = `h200_sxm.yaml` + `data/h200_sxm/`
(op 数据,可 symlink 内置)⊕ `data/h200_sxm/vllm/0.25.1/fpm_forward_perf.*`
(即采集产物;当前需手工拼 root,已提产品便利旗建议)。

```bash
# decode 单步(static_gen);三形分别:
$PY -m aiconfigurator.main cli estimate \
  --model-path MiniMaxAI/MiniMax-M2.7 --system h200_sxm --backend vllm \
  --perf-db-version 0.25.1 --systems-paths <root> \
  --forward-model fpm --estimate-mode static_gen \
  --tp-size 4 --moe-ep-size 4 --batch-size 32 --ctx-tokens 2048 --isl 2048 --osl 256
#   dep4: --tp-size 1 --attention-dp-size 4 --moe-ep-size 4
#   tp4:  --tp-size 4 --moe-tp-size 4
# prefill 单步(static_ctx):--estimate-mode static_ctx --batch-size 1 --isl 4096 --osl 1
```

实测输出(输入=上述参数,输出=下表):

| topo | prefill(bs1,isl4096) | decode(bs32,ctx2048) |
|---|---|---|
| tep4 | 9.45 seq/s(~106ms) | 59.70 tok/s/u,7.49 seq/s |
| dep4 | 14.54 seq/s 合计/4 副本(~275ms/副本) | 34.47 tok/s/u,17.30 seq/s(并发128) |
| tp4 | 9.01 seq/s(~111ms) | 75.26 tok/s/u,9.44 seq/s |

## 7. 产物存档(本仓库,scratch 外保全)

`artifacts/`:正式 parquet + 封印 metadata + 三份 run-manifest,
sha256 清单 `artifacts/SHA256SUMS.txt`。原始 benchmark JSON(25-52MB/cell)
留 scratch/PVC,sha 记录于各 run-manifest。

## 8. 已知边界(验收发现,修复在途)

- 缺陷1-3(prefix-caching pin / timeout / 校验器顺序断言):已修
  (8b5d4399/8552f522/68707ffb),随 #1475 走。
- 缺陷4:teleport 会话级流劣化致大件取件截断(sha 纪律响亮失败,resume
  可恢复;重登自愈);产品修法=分块传输/PVC 出件(开发 follow-up)。
- agg/disagg 全仿真在 fpm 模式下会查询采集包络之外(ramp 段 ctx<2/req),
  被 fail-closed 拒绝——上游 aic-core 需仿真 clamp 或网格补 ctx=1 档;
  static_ctx/static_gen 单步预测不受影响。
- 库根手工拼 systems 目录的摩擦;native 核须与代码同版本重编。
