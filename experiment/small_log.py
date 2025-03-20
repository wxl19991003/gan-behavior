import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子确保结果可复现
np.random.seed(42)

# 生成15个属性（attr0到attr14）
attr_ids = [f'attr{i}' for i in range(15)]  # 生成15个属性（attr0到attr14）

# 随机生成对应的注意力分布
attention_weights = np.random.rand(1, 15)  # 随机生成注意力分布

# 设置中文字体，防止中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置图形大小和其他参数
plt.figure(figsize=(10, 4))  # 增加宽度，使其不显得太瘦
sns.set(font_scale=1.2)  # 增加字体大小
sns.set_style("whitegrid")  # 设置背景为白色网格

# 绘制注意力分布热力图
sns.heatmap(attention_weights, 
            cmap='Blues',  # 使用之前的配色方案
            cbar_kws={'label': '注意力'},  # 给色条加标签
            xticklabels=attr_ids,  # 设置横坐标标签
            yticklabels=['注意力'],  # 设置纵坐标标签
            annot=True,  # 在每个格子内显示数值
            fmt='.2f',  # 设置显示数值的小数位数
            linewidths=0.5,  # 设置网格线宽度
            linecolor='gray')  # 网格线颜色

# 设置标题和标签
plt.title('注意力分布', fontsize=16)
plt.xlabel('属性', fontsize=12)
plt.ylabel('注意力', fontsize=12)
plt.xticks(rotation=45)  # 让横坐标标签有45度倾斜，避免重叠
plt.tight_layout()  # 自动调整布局以防止重叠
plt.show()
