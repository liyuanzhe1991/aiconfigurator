# R16 真值采集计划(C4)来源与裁定

- tep4_decode / tep4_prefill:逐字节复用 r15 已验证计划(l3_tep4_bundle/
  {decode_plan,prefill_plan}.csv)——新库覆盖空间与 r15 同界(C∈[1,1024],
  kv≤2.37M,产品网格实测同帽),复用使 MAPE 与 r15 基线的差异可归因于库,
  不混入计划几何变化。
- dep4_decode / dep4_prefill:同上,复用 l3_dep4_bundle。
- tp4_decode / tp4_prefill:tep4 计划同构复制(tp4 与 tep4 同为 tp=4/dp=1
  服务形,批量档与 kv 帽相同;产品采集网格实测两者 decode 均 1557 坐标)。
  tp4 为新 cell,首跑建档、不设判据(R16 §4)。
- 真值内容:ShareGPT 奇数池(采集偶数池,分池防考题污染);serving = parity
  配置 + 原版调度器,镜像 gc-timing-20260818(与采集同镜像)。
