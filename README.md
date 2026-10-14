# “Hey Cerebel” 唤醒词：模型 + 数据集（v1）

这个仓库里有：

| 文件夹 | 内容 |
|---|---|
| `model/wake.pt` | PyTorch 模型（TC-ResNet，约 2 万参数） |
| `model/wake_model.h` | 同一个模型导出成 C 数组，眼镜直接编译进 `cerebel-mic` |
| `model/report.txt` | 在没参与训练的说话人上的成绩 |
| `code/` | 生成数据、训练、导出、打包的全部脚本，以及眼镜端 C 推理（`wake.c`/`wake.h`）和对照测试 |
| `dataset/pos/` | 3299 条 “Hey Cerebel”（16 kHz 单声道 WAV） |
| `dataset/neg/` | 8544 条“不是唤醒词”：近音词（Hey Siri、Isabel、celebrate…）、日常句子、大模型写的 3000 句日常对话 |
| `dataset/manifest.csv` | 每条音频的标签、由哪个声音念的、该声音的许可、能否商用 |
| `dataset/negative_sentences.txt` | 负样本里大模型写的句子原文 |

## 怎么做出来的

- 全部是合成语音：Piper 开源语音合成的三个多人模型（LibriTTS-R 904 人、VCTK 109 人、L2-ARCTIC 24 个非母语说话人），加上 macOS 自带声音；每个说话人用不同语速、随机音色各念几遍。
- 训练时实时加噪音、房间混响、模拟眼镜小麦克风的频响、音量变化，还把半句 “Hey …” 当反例，避免听到 “Hey” 就触发。
- 输入：1.5 秒声音的 40 维 log-mel；眼镜上每 50 毫秒判断一次，一次不到 100 万次乘加。
- 眼镜上的 C 程序与 PyTorch 的分数逐次比对，误差在 0.000004 以内。

## v1 成绩（没参与训练的说话人）

| 阈值 | 认出 “Hey Cerebel” | 误触发 / 小时 |
|---|---|---|
| 0.8 | 93.1% | 53.5 |
| 0.9 | 89.1% | 30.9 |

误触发是在“故意塞满近音词”的最难测试上测的，日常说话会少很多；眼镜上还要求连续两次判断都过线，最后由云端再复核一遍。

## ⚠️ 商用许可

`manifest.csv` 的 `commercial_use` 一列标了每条能不能商用：

- **可以商用**（注明出处）：LibriTTS-R、VCTK 声音念的，CC BY 4.0。正样本 3039 条、负样本 5455 条。
- **不能商用**：
  - macOS 自带声音：苹果许可只允许个人、非商业使用。
  - L2-ARCTIC 声音：数据集是 CC BY-NC 4.0。
  - 共计正样本 260 条、负样本 3089 条。

**v1 模型用到了不能商用的数据，只能内部测试用。** 第二版正在只用可商用数据训练，上市请用第二版。以上许可是根据我对这些项目许可的了解整理的，正式上市前请法务核实。

## 下一步要补的数据

最能提升效果的是**真人用眼镜麦克风录的 “Hey Cerebel”**：不同人、不同距离、不同语气、有噪音的环境，每人 20～50 遍；再录几个小时戴着眼镜的日常对话做反例。

## 出处（CC BY 4.0 要求注明）

- 语音合成：[Piper](https://github.com/rhasspy/piper)（MIT），声音模型来自 [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)。
- LibriTTS-R：Koizumi et al., 2023，CC BY 4.0。
- VCTK：CSTR, University of Edinburgh，CC BY 4.0。
- L2-ARCTIC：Zhao et al., 2018，CC BY-NC 4.0（不可商用）。

## 用法

```bash
python code/gen_data.py data                  # 生成正样本和近音词反例（需要 piper-tts 和声音模型）
python code/gen_negatives.py data             # 让本地大模型写日常句子，再合成反例
python code/train.py data out --commercial    # 只用可商用数据训练
python code/export.py out test.wav            # 导出 C 权重，并输出对照分数
```

眼镜端：把 `wake_model.h` 放进 `cerebel_shell/include/`，重新编译 `cerebel-mic`（主仓库 `cerebel-ar-glasses`）。
