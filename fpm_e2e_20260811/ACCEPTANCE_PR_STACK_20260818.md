# PR 栈验收判决(2026-08-18,复核方=实验线 session)

对象:#1473 fpm-pr0-contract @82d83e50(delta 重构)→ #1475 fpm-prc-collector
@577824ab(#1474 已于 2026-08-11 被 squash 进 main,commit 1eedd536)。

## 判决:修复全部确认,满足两项 pre-merge 后放行合并(#1473→#1475)

审查方法:10-agent 对抗审查(逐修复簇验证 + 三套测试独立复跑 + 规则合规)
@5fb5b668;重构等同性 = collector/fpm_forward、fpm_contract.py 及全部 FPM
测试文件在 5fb5b668/82d83e50/577824ab 三方树哈希逐字节一致;终闸 =
577824ab 上 FPM 范围 308 测试全绿。

- B1(marker 类型比较):FIXED。database.py:91 str() 双边规范化;变异验证
  (撤修复→测试红)。残留:marker 声明侧建议注明须用 str() 规范形式。
- B2(resume+partial 吞失败):FIXED。runner.py:1740-1751 追加
  campaign_incomplete + :1767-1775 兜底分支;穷举确认仅剩两条 exit-0 路径
  (smoke 全过 / 全计划全过发布),四个对抗反例全符合预期。
- B3(smoke 可发正式库):FIXED。runner.py:1682 `not smoke` 臂 + branch1
  吸收全过 smoke;变异验证通过。注:测试真正的回归保护是精确分类断言
  (:2258/:2260),writer-stub raise 会被 except 吞,机制描述有偏差但净效果在。
- B4+M8(封印/first-wins 混合分支):FIXED。三测试齐且对守卫删除敏感;
  残留:corrupt-metadata 无用例、metadata-without-parquet 不对称(仅手删
  可达)、runner 层 skipped_first_publisher_wins 零覆盖 → follow-up。
- B5(resolved-config 入契约):FIXED。三方契约文件树哈希一致。
- admission②③④/C1/C2/M1/M2:FIXED。
- 测试:contract 49✓ / collector 827✓(6 skip 均 torch 缺席)/ generator
  289 passed + 4 skip(vs 声称 338 = 疑将契约 49 计入;4 skip 全在
  test_trtllm_extra_engine_args,待开发确认环境门控)。
- CI:#1473 全绿 MERGEABLE;#1475 除 Ruff 外全绿。

## pre-merge(两项,开发侧)

1. #1475 Ruff:collector/fpm_forward/runner.py:98:25(_utc_now,1 error,
   `--fix` 可修)。
2. docs/dynamo_deployment_guide.md 补 Guaranteed QoS 默认说明(规则审查
   NEW-in-increment 发现:999f4ddc 改所有 FPM pod 渲染清单而文档未动)。

## follow-up 批(随已计划的 ②env 透传 + manifest 相位树聚合 commit)

- flaky 加固:test_run_manifest_records_collector_phases_and_engine_interface
  (run_total_s 经 round(.,3),stub <0.5ms 归零挂断言;本机 20 连跑 0 失败,
  快机有暴露先例;修法=不 round 或假时钟)。
- planner.py:225-227 plumbing 断言补 file:line@version 引用(collector 规则)。
- FIXME(structural-marker) 与升级审计生命周期对齐(审计 grep 只认
  FIXME(kernel-limit)),连同规则修订提案 PROPOSAL_RULES_STRUCTURAL_ADMISSION.md
  等用户裁定。
- B4 残留三件(见上)。

## 记档不动(既定设计 / 用户已拍板)

- strict admission 第二过滤器(规则张力,提案挂账等裁定);
- marker 单侧校验(baseline_auto cell 无 marker 即免检);
- first-wins 静默丢新测行(exit 0 + checkpoint 字段,已记 failure-is-data 台账)。

## 合并执行

由实验线 session 执行,顺序 #1473 → #1475,前提 = pre-merge 两项落地 +
CI 复绿。generator follow-up 分支(Guaranteed QoS 独立评审单元)不开 PR,
等用户裁定——维持开发侧决定。
