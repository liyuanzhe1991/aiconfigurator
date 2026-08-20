# C1 终态设计提案:滤 fake + 上边缘有界线性外插(modeling 域)

> 所属 session:d07354d7(主 session)。用户 2026-08-20 裁定方向:去 fake 毒值后
> 用外插保住曲线尾部覆盖,不做 fail-closed 放弃。归属:fpm-modeling-dev,
> 基于分支 fpm-exclude-fake-fallback(b7e063f2)开新 PR。

## 动机与实证

1. fake_fallback 行的值不可信(4 卡虚高 ×2-3.7;2 卡实测虚低:tp2 b=128 顶格
   fake 33.24ms < real 档 41.51ms),但其所在 kv 带占真值负载不可忽略
   (tp2 同机真值 8.2% 票)。
2. 现状三种处置的同机真值实测(顶格带,b≤512):
   - 保留 fake 插值:tep4 29.8% / tp4 42.6% / tp2 11.9%(MAPE);
   - 删行 fail-closed(现实现):拒答不全,漏进错行匹配 bug,tp2 顶带 46-67%;
   - **删行 + 末两真实档线性外插:tep4 2.34% / tp4 0.34% / tp2 2.71%** ←采纳。
3. 错行 bug 可复现例:tp2,b=128,kv=131200,真值 41.69ms;滤后查询返回
   9.783885ms = 库中 b=2,kv=131072 行的值(行索引错位)。

## 实现三件套

1. **过滤时机前移**:装载器读行后、建网格前 retain(kv_seed_regime != "fake_fallback");
   先滤后建 → 阶梯自洽,错行 bug 结构性消失。默认开,逃生开关留对照。
   旧库(无列/全 null)逐位不变。
2. **kv 上边缘有界外插**:查询 kv > 滤后档顶时,末两真实档线性外推;
   上限 cap = 原始库(含 fake)该档顶格 kv,超 cap 拒答;真实行 <2 拒答;
   batch 维语义不动,外插值参与既有二维混合;结果带 extrapolated 标记。
3. **测试**:错行回归例(上述数字)、外插数学、cap 拒答、旧库不变、
   Rust/Python parity。

## 验收判据

- A/B 复现:三形同机顶格带 5,524 票,滤+外插 MAPE ≤ tep4 2.5 / tp4 0.5 / tp2 3.0;
- 零回归:网格内票逐位不变(tep4 同机滤后 4.17% 基准);
- 上游四闸(public-api/doctests/codeowners/DCO)。
- 验证脚本:同目录 tools_extrap_ab.py;数据:verify_samenode/*.csv.gz。
