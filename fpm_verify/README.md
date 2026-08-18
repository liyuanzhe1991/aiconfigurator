# fpm-verify — FPM 性能库对齐验证套件

独立于 aic 产品(用户裁定:verify 不属于 aic)。维护者:本仓库实验线。
用途:对 `aic fpm collect/build-db` 产出的性能库,用真实 serving 采收
ground truth 并做同坐标配对打分,给出 MAPE/P95/分层散点与对齐报告。

## 组成

| 目录 | 内容 |
|---|---|
| `drivers/` | 真值采收:decode 锁步收割驱动(`l3_decode_driver.py`)、prefill burst 驱动(`burst_driver.py`,含 DP 拦路石与四 rank 镜像门)、多端口遥测监听器、计划生成器、ShareGPT 奇数池内容源(`sharegpt_ids.py`) |
| `scoring/` | score-what-forms 配对打分:decode(`score_decode.py`,配速者合并/锁步窗口/F 层钳位)、prefill 单步格(`score_prefill_burst.py`)、长请求 chunk 补充口径(`score_prefill_chunks.py`) |
| `report/` | 散点(PNG,新旧库并排)与自包含 HTML 报告生成 |

## 口径要点(勿改动语义)

- 真值内容 = ShareGPT **奇数池**(采集用偶数池,分池防考题污染);
- serving 栈 = parity 配置 + **原版调度器**(stock 捕获),与采集同镜像;
- 打分 = 同坐标配对、一步一票、一坐标一票、跨窗中位、A/B/C 分层、
  配对弃行(任一侧查询失败两侧同弃);
- dep 拓扑 = 组内最大 wall(锁步语义)+ 镜像门(四 rank 同形 ±3%);
- 运行环境前提(运维责任):Guaranteed QoS pod、节点健康
  (参考 `../fpm_e2e_20260811/r15/clock_guard.sh` 自查)。

## 已知口径档案(打分时须知)

- dep4 prefill:产品范围外(2026-08-18 裁定),分数照出、不设判据;
- dep4 小步锁步服役 eager 主导(捕获态库低估 1.5-1.7×),档案见
  `../fpm_e2e_20260811/HANDOVER_R15_AIC_COLLECTOR.md`;
- launch-bound 带(每步 700-3000 token)对 CPU 配额敏感——truth 与
  collect 必须同为 Guaranteed pod,否则带内自带 ±8~18% 弥散。

## 状态

骨架期(自 r15 战役工装定版拷贝)。待办:统一 CLI 入口、参数外置
(模型/system/引擎脚本当前含 MiniMax/h200 特定值)、编排脚本模板化、
判据自动核对(R16 §4)。基准数字:decode 1.70/1.61%(tep4/dep4),
tep4 prefill 4.76%,详见 `../fpm_e2e_20260811/r15/R15_REPORT.md`。
