# FPM E2E 战役档案(2026-08-11,h200 / MiniMax-M2.7 / TEP4+TEP8 / 8k1k)

从"合并四个 FPM PR"到"发现并根因三类数据保真问题、验证一行修复、发布新采集镜像"
的完整证据链。跑法见 `../scripts/experiments/README.md`(runbook);本目录是结果与工件。

## 读什么

| 文件 | 内容 |
|---|---|
| `REPORT.md` / `REPORT.html` | **先读这个**:环境准备 → 五阶段流程(命令+结果)→ 验证方法与统计 → 待决事项 |
| `LEDGER.md` | 战役总账:来源钉板、每一笔预测-实测-归因、mixed 网格、根因判定、**Restart handoff(重启序列)** |
| `HANDOVER_MODELING.md` | **modeling session 从这里开始**:任务 A(mixed 公式)+ 任务 B(decode 批轴 regime 分区)执行书,含验收门与禁改清单 |
| `HANDOVER_BATCH_SEGMENT.md` | 批轴"段内括住"收尾改造(退役 k-NN)执行书,验收 oracle = `bracket_expected.csv` |
| `HANDOVER_DYNAMO_FPM.md` | dynamo-FPM 自基准改动全记录(randtok2):根因、salt 配对、镜像谱系、遗留三问题与上游化建议 |
| `L3_PLAN_V2.html` | **下一战役 L3 精度评测方案**:三分层测试集(A采集点/B离网/C外插)、矩阵生成器规范、指标(MAPE/P95/MAX APE+坐标)、工程四升级 |
| `MIXED_FORMULA_SPEC.md` | 公式推导全文 + 机制论证(交接书的依据) |

## 数据(全部可复算)

| 文件 | 内容 |
|---|---|
| `per_step_validation.csv` | 12,647 个真实引擎步的逐步对账(坐标+实测+模型值+delta) |
| `decode_validation_stack1.csv` | decode B×KV 窗级网格(B 1-1024 × KV 0.6k-200k/req) |
| `prefill_validation_stack2.csv` | prefill 全网格(20 token 档 × b1-4 + prefix 轴,配置匹配) |
| `mixed_validation_stack1.csv` / `_v2_cap2048.csv` | mixed chunk×Bd 网格(配置错配 v1 / 对齐 v2) |
| `probes/tep{4,8}/probe_r{1..5}.json` | 显式点位探针,全零输入 5 重复 |
| `probes/tep8/probe_rand*.json`, `probe_zero_small.json` | **判别实验**:随机 token vs 全零(同引擎同点位) |

## 关键结论(细节见 LEDGER)

1. **根因**:benchmark 合成输入是 `[0]*n` → MoE 路由退化 → 采集数据小 M 偏快 / 大 M 偏慢;
   一行随机化修复使 prefill 128-8192 对齐真实流量 ≤±1.5%(判别实验实测)。
2. **新采集镜像**:`nvcr.io/0980761089281446/dynamo-fpm-frozen:gc-steady-randtok2-20260812`。
3. **插值层达标**(v2 扩充探针,randtok2 数据:prefill 离网中位 1.1-2.8%,
   decode 离网中位 3.3-4.2%);**mixed 与 decode 批轴需公式改造**(见 SPEC §2/§5);
   **decode 余 -5~-9% 路由分布带**(dense 对照待做);3 条巨 KV 采集坏行待 QA 门。

复现脚本:`level0_closure.py`、`probe_analysis.py`、`per_step_validation.py`、
`make_probe_manifest.py`、`decode_sweep.sh`、`prefill_sweep.sh`、`mixed_sweep*.sh`、
`drive_probes*.sh`、`probe_exec.sh`。原始 FPM 流(数 MB)未入库,在本机 `serve_results/`。
