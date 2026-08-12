# Handover — decode 批轴"段内括住"改造(退役 k-NN)

交给 modeling-dev 执行。前置:#1461 的任务 A/B 已落地(query_totals + 粗分区),
本件是批轴解析的收尾升级。执行前读一遍 §0 的预期收益定位,别按"精度大招"来做。

## 0. 定位(诚实版)

**这不是精度大招,是构造性正确改造。** 离线 oracle 已算过全部 34 个离网探针点
(`bracket_expected.csv`):整体 MAPE 5.87% → 5.85%,几乎不动。原因:悬崖大头
已被 #1461 粗分区拿走;pad 区残余 +17~19%(tp8 b=12/20)是**行数据本身带路由
带偏低**,不是插值方法错。本改造的真实价值:

1. **消灭一类静默错误**:k-NN 跨 pad 段混票(b=8 的 8 档图便宜票投给 b=12)
   ——个别点可见改善((36,32760) tp8: +3.5%→+1.2%;(36,16380) tp4: +1.6%→+0.7%);
2. **假设与物理对齐**:引擎批轴是分段阶梯,平滑 k-NN 的错误在未来数据/新模型上
   不可预测;括住式的误差有界且可解释;
3. **简化**:FPM decode 批轴的解析从"4 邻加权 util 迁移"降为"两点线性",
   子表内 k-NN 成为死路径。

已知两点微退(写进 PR 描述,勿惊):(12, 733992) tp4 +0.6%→+6.5%(k-NN 靠
b=8 的票歪打正着);(36, 1677708) tp8 +28.4%→+32%(振荡口袋,两种方法都无效,
见 LEDGER round-2)。

## 1. 边界(与 HANDOVER_MODELING.md §0 相同,重申关键)

- 只改 `aic-core/src/aiconfigurator_core/sdk/operations/fpm_forward.py` 与
  Rust `perf_database/fpm_forward.rs` + `operators/fpm_forward.rs`;
- **共享 resolver 一行不改**(Python `perf_interp/engine.py`、Rust
  `perf_database/perf_interp.rs`——Rust GEMM raw 表共用 `Resolver::ScatteredSites`);
- prefill 不动(曲线轴 bisect 本来就是括住式);GEMM 等真散点场不动。

## 2. 设计

### 2.1 段划分(建表时,数据驱动;#1461 的检测框架可复用)

- capture 档序列:lattice 中存在 (x, x+1) 对的 x(decode 表:1,2,4,8,16,…,512);
- 细段:(x_k, x_{k+1}],括号行 = {x_k+1, x_{k+1}}(都跑 x_{k+1} 档的图);
- eager 段:(B*, max_batch],括号行 = {B*+1, max_batch}(本数据 {513, 1024});
  B* 沿用 #1461 的 ≥2× 跳变检测。

### 2.2 查询解析(op 层,替换子表内 k-NN)

```python
def _query_decode(self, db, b, kv):
    if b in lattice:  return self._resolve(cell, (b, kv), db)   # own-site,现状不动
    lo, hi = segment_bracket(b)                                  # §2.1
    cov = lambda x: curve_min(x) <= kv <= curve_max(x)           # 覆盖守卫,见 §2.3
    if cov(lo) and cov(hi):
        y_lo = self._resolve(cell, (lo, kv), db)                 # 精确站点 → 引擎 own-site 路径
        y_hi = self._resolve(cell, (hi, kv), db)
        w = (b - lo) / (hi - lo)
        return y_lo + w * (y_hi - y_lo)                          # 线性混合在 op 层
    if cov(hi):  return self._resolve(cell, (hi, kv), db)        # 单边:向 pad 目标取值
    if cov(lo):  return self._resolve(cell, (lo, kv), db)
    raise PerfDataNotAvailableError(...)                          # 双边都不覆盖 = 域外,不外推
```

- 混合空间:线性 latency(段内 GEMM 形状固定、attention 随真实条数近线性);
- b > max_batch:维持 #1461 的域/frontier 语义,不新增行为;
- `query_pass_baseline` 的 kv-floor 查询同样走此路径(floor 处两端覆盖必然成立)。

### 2.3 覆盖守卫(必须做,否则静默漏回 k-NN)

`own_curve_coverage_fallback=True` 意味着:若 kv 超出某括号行自己的曲线范围,
引擎会把该站点当不存在、**悄悄掉回 k-NN 投票**。op 层必须在发子查询前自查
两端行的曲线覆盖(建表时缓存每行 curve_min/max),按 §2.2 的单边/域外分支处理。
验收里有专门的负例测试。

### 2.4 Rust 同构

`perf_database/fpm_forward.rs`(#1461 已有段检测与子表)把子表路由替换为
bracket 解析;`operators/fpm_forward.rs` 查询入口同步。Python/Rust 同一轮改完,
parity 锁住。

## 3. 验收(oracle 已备好,机械对表)

1. **逐点对 oracle**:`bracket_expected.csv`(34 个离网点,含括号行、预期值
   `pred`、预期偏差 `d_new`)。实现后 op.query 输出与 `pred` 列一致
   (同一数学,容差 float 级;探针实测 `live` 列用于复算 `d_new`);
2. **零漂移**:A 组锚点(格点命中)、全部 prefill 点、decode 精确站点——修前后
   逐位相同;
3. **覆盖守卫负例**:构造 kv 超出 lo 行曲线顶端的查询,断言走单边分支且
   絶不触发引擎 k-NN(可用 monkeypatch/计数器断言 `_resolve_scattered`
   的邻居迁移段未执行);
4. **整体статы**:离网 B 组 MAPE 与现状差 ≤0.5pp(预期 5.87→5.85);
   两个已知微退点在 PR 描述中列明;
5. **op-level 回归门**:GEMM(Rust ScatteredSites 共用方)+ 全部 Grid 算子
   既有测试零 diff;engine-step parity 全绿;
6. CI 清单:codeowners / import contract / public-api contract / workspace
   doctests / DCO;本地 `uv sync` 重建 native .so 后再跑 parity(仓里可能有
   stale .so;cargo 在 `/opt/homebrew/opt/rustup/bin`)。

## 4. 不要做的事

- 不要动共享 resolver 的 `nn_sites`/距离度量/覆盖回退语义(GEMM 在用);
- 不要给 pad 区残余 +17~19% 加任何补偿常数——那是行数据的路由带偏差,
  归 collector/数据侧(dense 对照未做前禁运);
- 不要试图"修"振荡口袋点((36,1.68M) 类)——引擎行为本身多档,数据无法表达,
  已文档化为插值地板;
- 不要顺手清洗坏行(tp8 (256,6.56M)=11.22 等 3 条)——collector QA 门的活。

## 5. 证据索引

| 文件 | 内容 |
|---|---|
| `bracket_expected.csv` | **验收 oracle**:34 离网点的括号行/预期值/预期偏差 |
| `probes_v2_scores.csv` | 现状基线(#1461 后)全点位 |
| `MIXED_FORMULA_SPEC.md` §5 最终设计 | 本改造的规范与教训("先按物理分段,段内用最笨的插值") |
| `probes_v2/tep{4,8}/*.json` | 探针原始数据(实测列的出处) |
| `LEDGER.md` round-2 | 坏行/振荡口袋判定(§0 两个微退点的背景) |
