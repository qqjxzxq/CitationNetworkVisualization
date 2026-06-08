import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 1. 基础配置
COLOR_PALETTE = ['#8B7E6F', '#B4C4D5', '#9E9E7E', '#A58B84', '#7E8B9E', '#D6DADB', '#4A453F', '#C2B49B']
BG_COLOR = '#F2F0E4'

# 确保路径正确
df = pd.read_csv('combined_data_updated.csv')

# 目标字段
target_cols = [
    'author_source',
    'overall_interpretability',
    'physics_fusion_depth',
    'robustness_eval',
    'research_level'
]

# 需要剔除 none 的特定字段列表
cols_to_filter = ['author_source', 'physics_fusion_depth', 'research_level']


# 2. 绘图函数：根据参数决定是否剔除 'none'
def plot_stacked_area_flexible(df, column, ax, colors, filter_none=False):
    # 基础清洗：转为字符串并去空格
    temp_df = df[[column, 'publication_year']].copy()
    temp_df[column] = temp_df[column].astype(str).str.strip()

    # 只有在 filter_none 为 True 时才执行剔除
    if filter_none:
        exclude_vals = ['none', 'nan', 'Missing', 'null', '无', 'None']
        # 不区分大小写匹配剔除项
        filtered_df = temp_df[~temp_df[column].str.lower().isin([x.lower() for x in exclude_vals])]
    else:
        filtered_df = temp_df

    # 分组统计
    data = filtered_df.groupby(['publication_year', column]).size().unstack(fill_value=0)

    # 如果该字段全是空或者只有一年数据，跳过平滑处理防止报错
    if data.shape[0] < 2:
        return

    # 归一化为百分比
    data_perc = data.divide(data.sum(axis=1), axis=0)

    years = data_perc.index.values
    categories = data_perc.columns
    y_values = [data_perc[cat].values for cat in categories]

    # 平滑插值
    x_smooth = np.linspace(years.min(), years.max(), 200)
    y_smooth = []
    for y in y_values:
        spl = make_interp_spline(years, y, k=3)
        y_smooth.append(np.clip(spl(x_smooth), 0, 1))

    y_stack = np.cumsum(y_smooth, axis=0)

    # 填充绘图
    last_y = np.zeros_like(x_smooth)
    for i, (cat, current_y) in enumerate(zip(categories, y_stack)):
        ax.fill_between(x_smooth, last_y, current_y,
                        label=cat,
                        color=colors[i % len(colors)],
                        alpha=0.85)
        last_y = current_y

    # 美化
    title_suffix = '(Filtered "None")' if filter_none else '(All Categories)'
    ax.set_facecolor(BG_COLOR)
    ax.set_title(f'Trend: {column} {title_suffix}', fontsize=12, pad=10, color='#4A453F', fontweight='bold')
    ax.set_xlim(years.min(), years.max())
    ax.set_ylim(0, 1)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=8)


# 3. 创建画布并循环绘图
fig, axes = plt.subplots(len(target_cols), 1, figsize=(12, 4 * len(target_cols)), facecolor=BG_COLOR)

for i, col in enumerate(target_cols):
    # 检查当前列是否在需要过滤的名单中
    should_filter = col in cols_to_filter
    plot_stacked_area_flexible(df, col, axes[i], COLOR_PALETTE, filter_none=should_filter)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.xlabel('Publication Year', fontsize=12)
plt.show()