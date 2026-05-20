# 多模态睡眠调节算法框架技术整理

## 1. 项目目标

基于用户睡前及入睡早期的多模态生理数据，构建一个以 `PPO` 为核心的闭环控制系统，动态调节：

- 光源
- 香氛
- 温度

目标是：

- 缩短入睡时延
- 提升睡眠深度
- 降低微觉醒和夜间唤醒
- 在安全约束下实现个体化调节

## 2. 当前技术框架

当前推荐采用四层架构，而不是纯端到端强化学习。

### 2.1 总体分层

1. 感知层
2. 状态理解层
3. 决策控制层
4. 反馈学习层

### 2.2 感知层

负责采集并预处理用户睡前和睡眠早期的多模态输入。

核心输入：

- 心率 `HR`
- 呼吸率 `BR`
- 体温 `Temp`

建议补充：

- 心率变异性 `RMSSD/HRV`
- 呼吸不规则度 `BR_irregularity`
- 体动/静止度 `stillness`
- 睡前主观压力或认知负荷

这一层的主要任务：

- 时间同步
- 去噪与异常值剔除
- 滑动窗口聚合
- 构造相对 7 日基线偏差

### 2.3 状态理解层

这一层的核心是睡前状态分型器。

当前分型标签沿用定义书：

- `S1` 高交感/焦虑
- `S2` 高觉醒/亢奋
- `S3` 中度反刍
- `S4` 平静基准态
- `S5` 低唤醒/疲惫
- `S6` 病理性失眠风险

当前建议采用“两阶段状态识别”：

1. 规则分型器
2. 监督学习分型器

其中：

- 规则分型器用于冷启动和低数据阶段
- 监督学习分型器用于数据积累后的概率化升级
- 若置信度低，则回退到 `S4`
- `S6` 保持规则硬约束，不建议完全交给模型学习

输出结果：

- `state`
- `confidence`
- `rationale`

### 2.4 决策控制层

这一层采用“专家锚点 + PPO 微调”的控制结构。

控制逻辑：

```text
最终动作 = 专家锚点动作 + PPO输出偏移量
```

其中专家锚点动作由：

- 当前分型状态
- 当前睡眠阶段

共同决定。

PPO 不直接从零生成所有控制参数，而是在安全边界内学习微调量，这样更稳定、更可解释。

### 2.5 动作空间

当前动作空间主要包含三类控制量：

- 光源控制
- 香氛控制
- 温度控制

建议建模为连续动作：

```text
a_t = [
  light_lux_delta,
  aroma_level_delta,
  temp_delta_c
]
```

后续可以扩展为：

- 光节律频率
- 香氛脉冲占空比
- 渐暗速度
- 枕温/床温分离控制

### 2.6 安全约束层

睡眠调节属于强安全场景，必须加入显式约束。

核心安全机制：

- 光照上限约束
- 温度上下限约束
- 单步变化率约束
- 香氛累计剂量约束
- `S6` 进入保守策略或医疗提示

当前框架中，安全层位于 PPO 输出之后、设备执行之前。

### 2.7 奖励学习层

奖励函数负责评价当前策略是否帮助用户进入更好的睡眠状态。

当前采用双层奖励：

1. 即时稠密奖励 `dense reward`
2. 单晚终局奖励 `terminal reward`

即时奖励基于生理信号是否向“放松和入睡”方向变化：

- 心率是否下降并趋稳
- 呼吸是否更平稳
- 体温是否进入舒适区
- 体动是否减少
- 设备动作是否抖动
- 是否触发安全处罚

终局奖励基于睡眠结果：

- 入睡时延 `SL`
- 睡眠深度 `deep sleep ratio / proxy`
- 夜间觉醒次数
- 微觉醒次数

### 2.8 训练与部署框架

当前建议的完整训练闭环：

1. 规则策略上线采集数据
2. 构建状态分型数据集
3. 构建环境模型或仿真器
4. 离线训练 PPO
5. 小流量在线评估
6. 周期性个体化微调

推荐不要直接让 PPO 在真人真实环境中强探索。

## 3. 当前模型组成

结合当前代码框架，可以把模型系统拆成以下几个模块：

### 3.1 状态分型器

输入：

- `HR`
- `BR`
- `Temp`
- 可选 `RMSSD`
- 可选 `BR_irregularity`
- 可选 `stillness`
- 7 日基线信息

输出：

- 用户状态标签 `S1-S6`
- 置信度
- 判别依据

### 3.2 观测构造器

负责把原始生理数据和上下文特征拼接成 PPO 观测向量。

典型观测包括：

- 当前生理值
- 对基线偏差
- 当前阶段
- 当前状态标签
- 状态置信度
- 上一次动作
- 累计时间

### 3.3 锚点动作表

根据：

- 状态 `S1-S6`
- 阶段 `PRE_SLEEP / INDUCTION / N1_N2 / N3 / REM`

输出标准干预参数。

### 3.4 PPO 策略网络

输入：

- 当前观测向量

输出：

- 光调节偏移
- 香氛调节偏移
- 温度调节偏移

职责：

- 在专家锚点基础上做个体化细调

### 3.5 安全层

输入：

- 锚点动作 + PPO 偏移动作

输出：

- 满足边界约束的安全动作

### 3.6 奖励计算器

输入：

- 当前生理数据
- 用户基线
- 当前动作
- 上一动作
- 单晚结果

输出：

- `dense_reward`
- `terminal_reward`
- 总奖励

## 4. 所需数据类型整理

为了支撑当前算法框架，数据建议分成九类。

### 4.1 实时生理数据

这是最核心的数据类型。

字段建议：

- `timestamp`
- `user_id`
- `hr_bpm`
- `br_bpm`
- `skin_temp_c`
- `rmssd_ms`
- `br_irregularity`
- `stillness`

说明：

- `HR/BR/Temp` 是最低必需项
- `RMSSD`、`BR_irregularity`、`stillness` 是强建议项

### 4.2 用户基线数据

用于构造“相对个体偏差”，而不是只看绝对值。

字段建议：

- `baseline_date`
- `user_id`
- `hr_baseline`
- `br_baseline`
- `temp_baseline`
- `rmssd_baseline`
- `expected_sleep_hr`
- `expected_sleep_br`
- `preferred_temp_c`
- `avg_sleep_latency_7d`
- `abnormal_nights_7d`

### 4.3 用户上下文数据

用于解释同样生理值背后的不同状态来源。

字段建议：

- `age`
- `sex`
- `sleep_schedule_type`
- `exercise_within_3h`
- `caffeine_within_6h`
- `stress_score`
- `mood_score`
- `room_env_temp`
- `room_env_humidity`
- `weekday_or_weekend`

### 4.4 睡眠阶段数据

用于区分当前控制处于哪个阶段。

字段建议：

- `phase`
- `phase_start_time`
- `phase_end_time`
- `is_sleep_onset_detected`

阶段建议：

- `PRE_SLEEP`
- `INDUCTION`
- `N1_N2`
- `N3`
- `REM`
- `WAKE`

### 4.5 干预动作数据

这是 PPO 训练最关键的行为数据。

字段建议：

- `timestamp`
- `user_id`
- `phase`
- `state`
- `anchor_light_lux`
- `anchor_aroma_level`
- `anchor_temp_delta_c`
- `ppo_light_delta`
- `ppo_aroma_delta`
- `ppo_temp_delta_c`
- `final_light_lux`
- `final_aroma_level`
- `final_temp_delta_c`

### 4.6 设备执行反馈数据

用于确认“下发动作”和“实际执行”是否一致。

字段建议：

- `device_id`
- `command_timestamp`
- `actual_exec_timestamp`
- `actual_light_lux`
- `actual_aroma_level`
- `actual_temp_delta_c`
- `device_status`
- `command_success`

### 4.7 奖励与结果数据

用于训练 PPO 和做离线评估。

字段建议：

- `sleep_latency_min`
- `deep_sleep_ratio`
- `deep_sleep_proxy`
- `wake_count`
- `micro_arousal_count`
- `waso_min`
- `morning_subjective_score`
- `dense_reward_sum`
- `terminal_reward`
- `episode_total_reward`

### 4.8 标签与训练数据

用于训练分型器和辅助模型。

字段建议：

- `state_label`
- `state_confidence`
- `sleep_depth_label`
- `sleep_depth_proxy_label`
- `sleep_onset_label`
- `medical_flag`

标签来源可以包括：

- 专家规则标签
- 人工复核标签
- 睡眠问卷
- 多夜统计回标

### 4.9 安全与异常事件数据

用于安全评估和约束训练。

字段建议：

- `safety_violation_flag`
- `violation_type`
- `near_limit_flag`
- `rollback_flag`
- `manual_stop_flag`
- `device_fault_flag`

## 5. 当前最低可行数据集

如果现在要尽快跑通一个最小版本，最低可行数据集建议包含：

- `timestamp`
- `user_id`
- `hr_bpm`
- `br_bpm`
- `skin_temp_c`
- `stillness`
- `phase`
- `light_lux`
- `aroma_level`
- `temp_delta_c`
- `sleep_latency_min`
- `deep_sleep_proxy`
- `wake_count`

在这个版本下：

- 状态分型器可以先用 reduced-feature 模式
- 奖励函数可以先使用 `HR/BR/Temp + stillness + sleep_latency + deep_sleep_proxy`
- PPO 先学小范围动作偏移

## 6. 推荐数据结构格式

工程上建议采用三层数据表。

### 6.1 流式时序表

每 30s 或 60s 一条：

- 生理数据
- 上下文数据
- 动作数据
- 设备反馈

### 6.2 单晚 episode 表

每晚一条：

- 当晚主状态
- 入睡时延
- 深睡代理
- 奖励总分
- 是否异常

### 6.3 用户基线表

每个用户一条滚动更新：

- 7 日基线
- 常态睡眠指标
- 个体偏好温度
- 风险统计

## 7. 对当前项目的直接结论

目前这套模型最适合按下面方式推进：

1. 先用 `HR + BR + Temp + stillness` 跑通最小闭环
2. 状态分型器先使用规则版和 reduced-feature 降级逻辑
3. 奖励函数先采用“稠密奖励 + 单晚终局奖励”
4. PPO 只学习锚点动作周围的小偏移量
5. 数据积累后，再升级到监督分型器和更完整的 Safe PPO

## 8. 一句话总结

这套系统本质上是一个“多模态生理感知 + 状态分型 + 安全约束强化学习 + 睡眠结果反馈”的闭环睡眠调节框架。
