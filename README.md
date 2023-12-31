# Ink-Diffusion·墨迹——基于ChatGPT的儿童文学创作助手

## 背景调研

近年来我国家庭教育支出不断攀升，然而青少年心理问题也愈发普遍。据《中国科学院心理研究所国民心理健康数据库2022心理健康蓝皮书》显示，我国青少年抑郁风险检出率居高不下，需要引起社会广泛的重视。
<div align="center">
<img src="https://s2.loli.net/2023/09/08/5wdMHfhTVAQlqPy.png" width="728" height="320" />
</div>
通过对《语文》学科课外补习班部分**疑似**存在心理问题的儿童以及家长的访谈调研，我初步认定为当前环境下儿童出现不自信、表达欲丧失等表征的原因与家长“赶鸭式”、急功近利的教育方式有关。为了提振儿童在文学方面的创作欲望，我针对当前的部分痛点开发了Ink-Diffsion·墨迹，以丰富儿童文学创作产出为导向，使用GPT、DALL·E等API接口提供多模态的产出方式。在激发儿童创作兴趣的同时满足面向家长的期望产出。

## 功能概述

* InkStory：故事创作
  * 儿童自主选择故事走向
  * 刷新获取更多选项
  * “Publish”一键发布

* ToolBox：工具箱
  * “GPT Assistant”辅助完善语句，共10个档位循序渐进
  * 一键生成文生图提示词（基于近五年丰子恺儿童图画书奖获奖作品风格微调）
  * DALL·E2文生图工具生成绘本

## 界面展示

<div align="center">
<img src="https://s2.loli.net/2023/09/08/P2vLN8I5EtoeUcw.png" />
<img src="https://s2.loli.net/2023/09/08/cPjfeGuAQSOEt3x.png" />
<img src="https://s2.loli.net/2023/09/08/kqfenGK4T9LcZy6.png" />
<img src="https://s2.loli.net/2023/09/08/PzFwHxkieLmolh6.png" />
</div>

## 未来展望

* 使用国产大模型替代当前使用的**gpt-35-turbo-16k**
* 对文生图部分的模型调用从提示词升级为微调模型
* 整合gradio的audio和video组件，完成视频发布功能
