# fpm-verify 运行手册(端到端)

前提(运维责任):Guaranteed QoS pod(模板由 generator 渲染)、节点健康、
与采集同镜像。所有文件传输走 exec-cat + 双端 sha256。

## 1. 真值采收(pod 侧)

### 1.1 decode(锁步收割)
```bash
# 原版调度器 + parity serve 栈 + 多端口遥测监听
python drivers/make_l3_plan.py --topo <tep4|dep4> --out l3_plan.csv   # 计划:均匀格点⊕低kv填洞⊕锁步浅池
python drivers/l3_decode_driver.py l3_plan.csv <stream.jsonl> <windows.tsv>
```

### 1.2 prefill(burst 同拍)
```bash
# env: L3_MODEL_ID=<hf_id>  L3_DP_MODE=0|1  L3_DP_SIZE=<n>
python drivers/burst_driver.py prefill_plan.csv <stream.jsonl> <burst_windows.tsv>
# dep 拓扑自动启用 DP 拦路石 + 四 rank 镜像门(同形 ±3% 才入账)
```
真值内容一律 ShareGPT 奇数池(drivers/sharegpt_ids.py),与采集(偶数池)分池。

## 2. 打分(本地)

```bash
python scoring/score_decode.py        --topo <t> --stream ... --windows ... \
  --new-root <新库root> --old-root <旧库root> --out scores_decode.csv \
  --model-id <hf_id> --system <sys> --backend vllm --backend-version <v>
python scoring/score_prefill_burst.py  --topo <t> ...同上... --out scores_prefill.csv
python scoring/score_prefill_chunks.py --topo <t> ...(长请求补充口径,可选)
```
口径:同坐标配对、一步一票一坐标一票、跨窗中位、A/B/C 分层、配对弃行、
dep 配速者合并 + 锁步组 wall=max。

## 3. 报告

```bash
python report/make_scatter.py --scores scores_prefill.csv --title ... --out scatter.png
python report/gen_report.py --manifest verify_manifest.json   # 清单驱动,格式见脚本头
```

## 4. 判据(R16 §4)

decode ≤2.0%(基准 1.70/1.61%);tep4 prefill ≤5.5%(基准 4.76%);
dep4 prefill 范围外照出不判;新 cell 首跑出数记档。

## 待办(维护者)

- 编排脚本模板化(serve 拉起/监听器/驱动的 pod 侧序列,现散于
  fpm_e2e_20260811/r13、r15 的 sequencer,需收编);
- drivers 内少量常量外置(vocab 上限、budget);
- 判据自动核对(读分数 CSV 对照 R16 §4 直接出 PASS/FAIL)。
