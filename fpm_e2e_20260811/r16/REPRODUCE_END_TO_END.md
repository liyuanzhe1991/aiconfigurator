# 端到端复现走廊:从 collect 到 verify(单线,可照抄)

> 目标读者:没有本战役任何上下文的人。沿本文从上到下走一遍 =
> 完整复现"采集 → 建库 → 预测 → 独立验证打分"。每步给出字面命令、
> **成功标志**(checkpoint)与失败排查指针。深挖细节的入口:
> 采集侧 [`R16_RUNBOOK.md`](R16_RUNBOOK.md),验证侧
> [`VERIFY_METHODOLOGY_R15_R16.md`](VERIFY_METHODOLOGY_R15_R16.md)。
> 所属 session:`d07354d7-72f0-47a1-aeab-be8fd346e942`。

## 第 0 步:前置(一次性)

**先检出正确的代码树(最重要的一步)**:本档案分支(fpm-all-20260811)
的 collector 代码是旧基线,**直接在其上跑采集会缺 run-manifest、tp4 会
无感复刻已定罪的 fake 制度数据**。产品代码在:

```bash
git checkout b1-kv-seed-regime   # 推荐:含 R16 三修复(至 68707ffb)+ A2 渲染(defcc285)+ B1 列(f2a2b61b)
# 仅需逐位复刻 R16 战役当日行为时:git checkout 68707ffb
```

```bash
# 集群凭证(SSO 浏览器授权;>4h 长任务中大文件传输变慢时重新 login)
tsh login --proxy=nv-prd-dgxc.teleport.sh
tsh kube login dynamo-nebius-2          # H200 集群

# 仓库环境(仓库根执行)
# venv(一次):
#   python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" -e ./aic-core
PY=$PWD/.venv/bin/python
export PYTHONPATH=$PWD/src:$PWD/aic-core/src
export FPM_KUBECTL="kubectl --context=nv-prd-dgxc.teleport.sh-dynamo-nebius-2"
# native 核(仅打分/预测需要;与代码版本必须同步):
#   cd aic-core && uvx maturin build --release   # cargo 在 /opt/homebrew/opt/rustup/bin(不在默认 PATH)
#   wheel 解出 _aiconfigurator_core.abi3.so 放 aic-core/src/aiconfigurator_core/
```

**环境专名注意**:命令中的 namespace `yuanli-aic`、PVC `model-cache`、
secret `nvcr-push-secret`、队列 `dynamo` 均为本环境专名,外部环境请替换;
资产缺失时的上载流程本文不覆盖。

前置资产(共享 PVC `model-cache` 上应已存在):模型 snapshot
`models--MiniMaxAI--MiniMax-M2.7/snapshots/d494266a…`、ShareGPT 数据集
`fpm_datasets/ShareGPT_V3_unfiltered_cleaned_split.json`。

**镜像选择(关键)**:tep/dep 拓扑用 `gc-timing-20260818`;**tp4(pure_tp)必须用 `gc-warmtp-20260820`(digest sha256:2cc6444956a6…e91424)
或 `gc-vocabfix-20260820`(双修,推荐;digest sha256:b2ca3f8a…c777)**,并且
代码树含 A2 渲染改动(即第 0 步检出的 b1-kv-seed-regime)——否则 tp4 decode 采出的是 fake
制度数据(已定罪偏差 −15.3%,详见 RUNBOOK §8)。

✅ 成功标志:`$FPM_KUBECTL get pods -n yuanli-aic` 能列出(可为空);
`$PY -c "import aiconfigurator_core"` 无报错。

## 第 1 步:采集(collect;先冒烟后正式)

公共尾部 `<TAIL>` 见 [`R16_RUNBOOK.md`](R16_RUNBOOK.md) §3(模型缓存、
镜像、调度器标签、`--fpm-database-root` 等,逐字照抄;tp4 记得把镜像
tag 换成 gc-warmtp-20260820)。

```bash
# 冒烟(不写库,~25 分钟)
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --fpm-max-gpus 4 --fpm-parallel-presets tep --fpm-tp-sizes 4 <TAIL 去 database-root> --smoke --limit 2

# 正式(三形各一跑,串行;三形合计 ~4h(预热 46%);tep4 单形 ~1.9h,其 decode cell 预热占 81%)
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --fpm-max-gpus 4 --fpm-parallel-presets tep     --fpm-tp-sizes 4 <TAIL>
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --fpm-max-gpus 4 --fpm-parallel-presets dep     --fpm-dp-sizes 4 <TAIL>
$PY collector/collect.py --backend vllm --ops fpm_forward \
  --fpm-max-gpus 4 --fpm-parallel-presets pure_tp --fpm-tp-sizes 4 <TAIL>
```

**采集开跑后立刻做一件事(同机协议的前半)**:记下每个 cell pod 的节点名
——aic 不记录,这是验证方自己的责任:

```bash
$FPM_KUBECTL get pods -n yuanli-aic -o wide | grep fpm-   # 记下 NODE 列
```

✅ 成功标志:collect 进程结尾打印 COLLECTION SUMMARY 且 `Total errors: 0`;
`fpm_formal_database/.../fpm_forward_perf.parquet` 出现且伴随
`.metadata.json` 封印(parquet_sha256/row_count);
`fpm_forward_artifacts/<run>/run-manifest.json` 存在。
❌ 排查:cell 失败看 `fpm_forward_artifacts/<run>/cells/<cell>/logs/`;
错误分类在 `all_<时间戳>/collection_summary_vllm.json`。

## 第 2 步:建库与预测(build 内建于 collect;预测验通)

建库无独立命令——collect 成功即已发布进 `--fpm-database-root`
(sha 封印 + first-publisher-wins 合并)。用一条预测命令验通链路:

```bash
$PY -m aiconfigurator.main cli estimate \
  --model-path MiniMaxAI/MiniMax-M2.7 --system h200_sxm --backend vllm \
  --perf-db-version 0.25.1 --systems-paths <root> \
  --forward-model fpm --estimate-mode static_gen \
  --tp-size 4 --moe-ep-size 4 --batch-size 32 --ctx-tokens 2048 --isl 2048 --osl 256
# <root> 拼装(字面):
#   mkdir -p /tmp/fpmroot/data/h200_sxm
#   cp src/aiconfigurator/systems/h200_sxm.yaml /tmp/fpmroot/
#   ln -s $PWD/src/aiconfigurator/systems/data/h200_sxm/* /tmp/fpmroot/data/h200_sxm/ 2>/dev/null
#   mkdir -p /tmp/fpmroot/data/h200_sxm/vllm && ln -s $PWD/fpm_formal_database/data/h200_sxm/vllm/0.25.1 /tmp/fpmroot/data/h200_sxm/vllm/0.25.1
# 三形参数差异见 RUNBOOK §6(dep4: --tp-size 1 --attention-dp-size 4)
```

✅ 成功标志:输出吞吐数字量级合理(tep4 decode bs32/ctx2048 ≈ 60 tok/s/u
一带,参照 RUNBOOK §6 实测表)。

## 第 3 步:独立验证——真值收割(verify/harvest)

```bash
# 同机协议(必须):PIN_NODE = 第 1 步记下的采集节点
PIN_NODE=<采集节点名> bash fpm_verify/harvest/stage_and_run.sh tep4 nv-prd-dgxc.teleport.sh-dynamo-nebius-2
```

kit 自动完成:部署 4×H200 pod(钉死同节点)→ 时钟守卫(<1900MHz 整链
失败并提示拉黑节点)→ 文件 staging(双端 sha)→ 单 nohup 链跑三相
(serve 栈 → decode 锁步窗 → prefill burst + mixed)。tep4 全程 ~4-5h。

**真值驱动版本注意**:kit 当前默认 decode 驱动为 ShareGPT 版(v2);
若要与 R16 战役的历史数字逐位可比,用 `L3_DECODE_DRIVER=bench` 回退旧
随机内容路径(详见方法论 §1.2 版本注意框)。

✅ 成功标志:pod 内 `/results/phase_all.log` 出现 `HARVEST-ALL-DONE`;
`/results/nodeName.txt` = 你钉的节点。
❌ 排查:`fpm_verify/harvest/README.md` 故障恢复节(引擎家族清杀清单、
显存排空、驱动熔断语义)。

## 第 4 步:取件(分块 sha + 断点续传)

```bash
bash fpm_verify/harvest/fetch_results.sh tep4 nv-prd-dgxc.teleport.sh-dynamo-nebius-2 <本地目录>
```

✅ 成功标志:输出每文件 `OK`,`<本地目录>/SHA256SUMS.txt` 生成;
中断重跑同一命令即断点续传。

## 第 5 步:打分(同真值配对两库)

```bash
FPM_EXCLUDE_FAKE_FALLBACK=1 PYTHONPATH=$PWD/src:$PWD/aic-core/src $PY \
  fpm_verify/scoring/score_decode.py --topo tep4 \
  --stream <本地目录>/fpm_stream.jsonl --windows <本地目录>/decode_windows.tsv \
  --new-root <被验库根> --old-root <基线库根> --out scores.csv
# prefill 层:score_prefill_burst.py --stream/--windows 换 burst 文件,参数同形
# dep4/tp4:--topo 换名;dep4 新库自动用原生身份(tp1/dp4/ep4)
```

`--new-root/--old-root` 均须是第 2 步的"系统根"结构(不能直接指
fpm_formal_database/);基线库根可用本仓库存档
`fpm_e2e_20260811/r16/artifacts/` 里的 parquet 照第 2 步方式拼一份。

`FPM_EXCLUDE_FAKE_FALLBACK=1` 的作用:排除库中已知不可信的顶格回退行
(需库带 kv_seed_regime 列,B1 之后的采集自带;老库无列时空转无害)。

✅ 成功标志:stdout 打出 old/new 两侧的分层 MAPE 表;
`真值坐标 n 万个` 量级(tep4 decode ~8 万)。

## 第 6 步:判读(数字什么时候算 PASS)

- **同机口径**(真值 PIN 在采集节点):decode ≤2.0%、tep4 prefill ≤5.5%;
- **跨机口径**(没钉同机):±3-4% 是健康节点间的天然地板,decode 判据
  放宽为 ~5% 且只作参考——不要用跨机读数下"库退化"的结论;
- 参照系:终审记分牌(FINAL_REPORT §0.4 与第一章;交互版图表在 .html 第 7 章)——tep4 decode 同类跑法
  预期 ~3.7-4%(跨机)/ ~2% 级(同机),tp4(warmtp/vocabfix 镜像采集)预期 ~4.8%(跨机;残差主要为跨机地板 ±3-4% 与小批段节点账)。

## 常见坑速查(全部实战踩过)

| 症状 | 原因与解法 |
|---|---|
| tp4 decode 分数 ~10% | 用了旧镜像(fake 制度)——换 gc-warmtp-20260820 + A2 渲染 |
| 真值全体均匀偏慢 ~7% | 锁频病卡——时钟守卫会拦;拉黑该节点重跑 |
| 分数比预期差 ~3-6% 且集中小批量 | 跨机比较——回去补同机(PIN_NODE) |
| 大文件取件中途断 | teleport 会话劣化——重新 tsh login 后重跑 fetch(自动续传) |
| 打分变慢 20 倍 | 手改过 parquet 动了网格——不要删行;过滤交给 env 开关 |
| 巨 kv 段尖刺(MAX >30%) | 没开 FPM_EXCLUDE_FAKE_FALLBACK 或库无列(老库) |
