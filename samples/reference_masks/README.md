# 完整掩模参考样本

positive/circle：圆形目标正样本。
positive/square：方形目标正样本。
negative/ellipse：椭圆干扰物负样本。

每个子目录包含 image.png（参考原图）、mask.png（同尺寸完整黑白掩模）及 provenance.json（来源）。
本次完整掩模沿用上一轮模型输出作为演示参考，来源已记录；不是独立人工标注或评估真值。
正式样本可用人工确认的完整掩模替换。白色为该样本的标注区域，黑色为背景。

取点及 SAM2 推理规则不变，目录名称和颜色不参与模型判别。
configs/*.references.json 引用样本，build 后再运行 configs/pipeline.json，分别输出 circle.mask.png 和 square.mask.png。
上传的拼接图片只作格式示例，不作为本次原尺寸输入掩模。
