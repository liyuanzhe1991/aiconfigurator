## 使用方式
### 数据采集
* 采集脚本存放在 `test/hisim/collect` 目录，`bench_serving.py` 由 sglang 0.5.6.post2 的压测脚本修改而来，主要针对 gsp 的前缀压测进行修改
* 当前采集的数据在 `33.254.38.20:/apsarapangu/disk2/projects/kunlun_insight/data/`（容器内挂载NFS需要启动权限 `--cap-add=SYS_ADMIN`）
### 结果统计
* `replay.py`：用于完美回放测试
* `simulate.py`：仿真测试以及误差统计
    * `python ./simulate.py --data-dir serving/sglang/0.5.6.post2/260213-L1-H20-Qwen3-32B-FP8-GSP-No-Overlap/`
    * `--aic-db-path`：指定AIC数据的路径，可以设定环境变量 `KUNLUN_INSIGHT_COMMON_DATA_DIR` 指向NFS `projects/kunlun_insight/data`
    * `--runner`：`offline` 离线无 HTTP，`serving` 服务化方式压测（Hisim Readme方式）
    * `--cases`：遍历的Case名（文件名），如 `--cases "no_cache" "l1"`
    * `--output-dir`：输出路径名，默认为 `test/hisim/output/xxx`