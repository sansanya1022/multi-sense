# sleep_rl_controller

个体化睡眠调节算法 MVP 工程。

本项目面向睡前和入睡早期场景，基于用户年龄、正常生理基线和实时生理信号，动态调节光源、香氛、温度。系统采用严格的双层控制结构：

```text
final_action = personalized_anchor_action + tiny_model_delta
final_action = safety_layer(final_action)
```

不会使用端到端强化学习直接控制全部动作。

## 系统架构

系统由以下模块组成：

1. 用户画像与基线数据结构
2. 个体化参数生成器
3. 规则状态分型器，输出 S1-S6
4. 个体化锚点策略
5. 观测向量构造器
6. 端侧小模型 `TinyMLPPolicy`
7. 安全层 `SafetyLayer`
8. 控制器 `SleepController`
9. 奖励函数 `RewardCalculator`
10. 简单仿真环境 `SleepSimulator`
11. PPO / 蒸馏训练接口骨架
12. TorchScript / ONNX / int8 占位导出

## 为什么采用 anchor + delta

采用 `anchor + delta` 而不是端到端控制的原因：

- 专家规则和安全先验可直接进入系统
- 冷启动时无需依赖大规模训练数据
- 控制逻辑更可解释
- 更适合迁移到 ESP32-S3 / ESP-DL 端侧部署
- 便于对不同年龄与敏感度用户做个体化缩放

## 为什么 S6 不交给模型学习

`S6` 表示病理性失眠风险下的保守控制模式，不应由小模型自由探索。原因如下：

- 该状态有较高安全风险
- MVP 不是医疗诊断系统
- 该状态下更适合固定保守锚点策略和强安全边界

因此，`S6` 下模型 delta 被强制置零，仅执行保守锚点动作和安全层。

## 安装

推荐 Python 3.11。

```bash
pip install -r requirements.txt
```

## 运行 demo

在工程根目录执行：

```bash
python examples/demo_mvp.py
```

## 运行测试

```bash
pytest
```

## 运行结果

demo 会：

- 加载样例用户画像和基线
- 构造一条实时生理样本
- 调用控制器做一次决策
- 输出状态、置信度、锚点动作、模型微调、最终安全动作
- 运行一晚仿真并输出 `EpisodeOutcome`

## 工程结构

```text
sleep_rl_controller/
  README.md
  pyproject.toml
  requirements.txt

  sleep_controller/
    __init__.py
    schemas.py
    personalized_engine.py
    state_classifier.py
    anchor_policy.py
    observation_builder.py
    tiny_model.py
    safety_layer.py
    controller.py
    reward.py
    simulator.py
    training.py
    export.py
    utils.py

  examples/
    demo_mvp.py
    sample_user_profile.json
    sample_user_baseline.json

  tests/
    test_personalized_engine.py
    test_state_classifier.py
    test_observation_builder.py
    test_safety_layer.py
    test_controller.py
    test_reward.py
```

## 迁移到 ESP32-S3 / ESP-DL int8 的路径

后续迁移建议：

1. 保留 `PersonalizedEngine`、`StateClassifier`、`AnchorPolicy`、`SafetyLayer` 的规则逻辑
2. 将 `TinyMLPPolicy` 量化为 int8
3. 使用 `export.py` 导出 TorchScript / ONNX
4. 在 ESP-DL 中重建同结构 MLP，并加载量化权重
5. 设备端仅部署前向推理，不部署 PPO 训练逻辑

## 安全说明

本项目是睡眠环境调节算法原型，不是医疗诊断系统。

`S6` 仅表示保守控制模式，不用于诊断失眠或疾病。

算法输出仅用于舒适性环境调节，不应替代医生建议。

