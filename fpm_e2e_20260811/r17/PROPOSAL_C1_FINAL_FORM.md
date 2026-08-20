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

## 实现三件套(v2 修正,2026-08-20 用户点破:外插能力已内建,勿另造)

架构事实:fpm_forward 查询 = 外层域闸(超采集域硬拒,先于插值引擎)+
内层 perf_interp(域内 beyond-the-range 已有 util-hold 边界语义:持边界
利用率、按 SOL 解析模型缩放作答——自带 KV 带宽线性项的物理外推)。

1. **过滤时机前移**:装载器读行后、建域闸表与插值站点前统一
   retain(kv_seed_regime != "fake_fallback");先滤后建 → 两层结构一致,
   错行 bug(域闸/站点表不一致的产物)结构性消失。默认开,留逃生开关。
   旧库(无列/全 null)逐位不变。
2. **域闸 kv 上限参数化(零新数学)**:过滤后域闸天花板保持为原始
   (含 fake 行)顶格 cap,不随过滤收缩——"滤后顶格→cap"带的查询过闸后
   自然落入 perf_interp 现成 util-hold 语义,由 SOL 物理外推作答。
   不写新外插器;本提案附带的线性外插仅作验收基准线。
3. **测试**:错行回归例(上述数字)、域闸 cap 语义(cap 内作答/cap 外
   硬拒)、util-hold 在该带的作答走通、旧库不变、Rust/Python parity。

## 验收判据

- A/B 复现:三形同机顶格带 5,524 票,滤+util-hold 作答的 MAPE 对标线性外插基准(tep4 2.34/tp4 0.34/tp2 2.71),不劣于即收;
- 零回归:网格内票逐位不变(tep4 同机滤后 4.17% 基准);
- 上游四闸(public-api/doctests/codeowners/DCO)。
- 验证脚本:同目录 tools_extrap_ab.py;数据:verify_samenode/*.csv.gz。

## v3 增补(2026-08-20 晚,用户核对本意后的简化)

own_curve_coverage_fallback 的设计本意 = 保护"孤儿站点"(一两个散点的残缺
曲线)不自答远外推;实测 decode 三形(tp2/tp4/tep4)各 102 站点,曲线长度
滤后最短 6 点、中位 10-15 点,**孤儿站点数 = 0**——该开关在 decode 上没有
任何正当保护对象,只有误伤(b=128 借 b=2 案即其产物)。

修法②最终形态(一行):fpm_decode_config 关闭 own_curve_coverage_fallback
(decode 越界查询走自家曲线 util-hold 物理外推);fpm_prefill_config 保留
该开关(prefill 站点为 (batch,kv) 二维对,存在真孤儿)。前沿豁免收紧(修法③)
与打分工装接真 SOL(修法④)维持不变。

## v4 定稿(2026-08-20 晚,用户指出"先滤后建"信息扔了再捡的别扭,改为行分角色)

废弃"先滤后建 + 域闸 cap 参数化"(两处事实源,删行后又要捞回原顶格,蠢)。
终态设计 = **单次构建、行分角色**:
1. 全量行(含 fake)一次性构建全部结构——域闸天花板、覆盖、站点坐标照旧,
   零第二套结构,错行类不一致由构造保证不可能;
2. fake_fallback 行降级为"仅坐标行":参与定义域(坐标物理可达的凭证),
   **不进值锚点集**——曲线求值/插值/util-hold 只锚 real 行;
3. 查询落在 [末 real 档, fake 顶格] 带:域闸放行(fake 坐标撑天花板),
   求值走自家 real 尾部 util-hold 物理外推,fake 的值全程无人使用;
4. decode 配置关闭 own_curve_coverage_fallback(v3 结论,孤儿站点=0);
   prefill 保留;前沿豁免收紧与打分工装接真 SOL 维持。
语义即用户原始意图:"值不可信、坐标可信"。
