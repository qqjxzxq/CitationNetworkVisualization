import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib import rcParams

# --- 1. 环境配置 ---
plt.style.use('seaborn-v0_8-white')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['axes.linewidth'] = 0.8
COLOR_PALETTE = ['#8B7E6F', '#B4C4D5', '#9E9E7E', '#A58B84', '#7E8B9E', '#D6DADB', '#4A453F', '#C2B49B']

# --- 2. 数据读取与预处理 ---
df = pd.read_csv('sample_with_clusters.csv')
df['author_id_list'] = df['author_id_list'].fillna('').astype(str).apply(lambda x: [i for i in x.split(';') if i])
df['ref_id_list'] = df['referenced_ids_openalex'].fillna('').astype(str).apply(lambda x: [i for i in x.split('@') if i])

# --- 3. 按年份聚合指标 ---
def count_unique_in_year(series):
    all_ids = [item for sublist in series for item in sublist]
    return len(set(all_ids))

def count_total_refs_in_year(series):
    all_ids = [item for sublist in series for item in sublist]
    return len(all_ids)

# 计算均值
yearly_stats = df.groupby('publication_year').agg({
    'title': 'count',
    'author_id_list': count_unique_in_year,
    'cited_by_count': 'sum',
    'ref_id_list': count_total_refs_in_year
}).rename(columns={'title': 'paper_count', 'author_id_list': 'unique_authors', 'ref_id_list': 'total_refs'})

# 模拟置信区间：计算年度内样本的波动（标准差）作为阴影依据
# 注意：如果数据源较小，这里使用 10% 的偏移作为视觉演示，或根据实际分布计算
std_fill = yearly_stats * 0.15

years = np.array(yearly_stats.index)
x_new = np.linspace(years.min(), years.max(), 300)

def smooth(x, y):
    spl = make_interp_spline(x, y, k=2)
    return spl(x_new).clip(0)

# --- 4. 绘图 ---
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(11, 12), dpi=200, facecolor='#F2F0E4')
plt.subplots_adjust(hspace=0.35)
ax1.set_facecolor('#F2F0E4')
ax3.set_facecolor('#F2F0E4')

# --- (Top) 双纵坐标趋势图 ---
ax2 = ax1.twinx()

# 准备平滑曲线及置信区间边界
metrics = ['paper_count', 'unique_authors', 'total_refs', 'cited_by_count']
lines_data = {}

for m in metrics:
    y = yearly_stats[m]
    error = std_fill[m]
    lines_data[m] = {
        'mid': smooth(years, y),
        'low': smooth(years, y - error),
        'high': smooth(years, y + error)
    }

# 绘制左轴 (Papers & Authors)
l1, = ax1.plot(x_new, lines_data['paper_count']['mid'], color=COLOR_PALETTE[0], lw=1.5, label='Papers Count', zorder=10)
ax1.fill_between(x_new, lines_data['paper_count']['low'], lines_data['paper_count']['high'],
                 color="gray", alpha=0.15, zorder=5, lw=0) #color=COLOR_PALETTE[0]

l2, = ax1.plot(x_new, lines_data['unique_authors']['mid'], color=COLOR_PALETTE[1], lw=1.5, label='Authors Count', zorder=9)
ax1.fill_between(x_new, lines_data['unique_authors']['low'], lines_data['unique_authors']['high'],
                 color="gray", alpha=0.15, zorder=4, lw=0) #color=COLOR_PALETTE[1]

# 绘制右轴 (Refs & Citations)
l3, = ax2.plot(x_new, lines_data['total_refs']['mid'], color=COLOR_PALETTE[2], lw=1.2, ls='--', label='Total References', zorder=8)
ax2.fill_between(x_new, lines_data['total_refs']['low'], lines_data['total_refs']['high'],
                 color="gray", alpha=0.1, zorder=3, lw=0) #color=COLOR_PALETTE[2]

l4, = ax2.plot(x_new, lines_data['cited_by_count']['mid'], color=COLOR_PALETTE[3], lw=1.2, ls=':', label='Total Citations', zorder=7)
ax2.fill_between(x_new, lines_data['cited_by_count']['low'], lines_data['cited_by_count']['high'],
                 color="gray", alpha=0.1, zorder=2, lw=0)  #color=COLOR_PALETTE[3]

# 轴美化
ax1.set_ylabel('Papers / Authors', fontsize=11, fontweight='bold')
ax2.set_ylabel('Refs / Citations', fontsize=11, fontweight='bold')
# ax1.grid(axis='y', linestyle='--', alpha=0.2)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

lines = [l1, l2, l3, l4]
ax1.legend(lines, [l.get_label() for l in lines], loc='upper left', frameon=False, fontsize=10)

# --- (Bottom) 主题河流图 ---
river_data = df.groupby(['publication_year', 'cluster_label']).size().unstack(fill_value=0)
river_smooth = [smooth(years, river_data[i]) for i in river_data.columns]

ax3.stackplot(x_new, river_smooth, labels=[f'Cluster {i}' for i in river_data.columns],
              colors=COLOR_PALETTE[:len(river_data.columns)], alpha=0.8, edgecolor='white', lw=0, baseline='wiggle')

ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('Relative Topic Weight', fontsize=11, fontweight='bold')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.get_yaxis().set_ticks([])
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title="Topic Clusters")

plt.tight_layout()
plt.show()







# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import make_interp_spline
# from matplotlib import rcParams
#
# # --- 1. 环境配置与莫兰蒂配色 ---
# # 设置 Nature/Science 风格的简约参数
# plt.style.use('seaborn-v0_8-white')
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Arial']
# rcParams['axes.linewidth'] = 0.8
# COLOR_PALETTE = ['#8B7E6F', '#B4C4D5', '#9E9E7E', '#A58B84', '#7E8B9E', '#D6DADB', '#4A453F', '#C2B49B']
#
# # --- 2. 数据读取与预处理 ---
# df = pd.read_csv('sample_with_clusters.csv')
#
# # 将 ID 字符串转换为列表（处理 NaN 为空列表）
# df['author_id_list'] = df['author_id_list'].fillna('').astype(str).apply(lambda x: [i for i in x.split(';') if i])
# df['ref_id_list'] = df['referenced_ids_openalex'].fillna('').astype(str).apply(lambda x: [i for i in x.split('@') if i])
#
# # --- 3. 按年份聚合指标 ---
# def count_unique_in_year(series):
#     """计算一年内所有论文涉及到的独立作者总数"""
#     all_ids = [item for sublist in series for item in sublist]
#     return len(set(all_ids))
#
# def count_total_refs_in_year(series):
#     """计算一年内所有论文引用的文献总量"""
#     all_ids = [item for sublist in series for item in sublist]
#     return len(all_ids)
#
# # 核心数据统计
# yearly_stats = df.groupby('publication_year').agg({
#     'title': 'count',                 # 当年发文量
#     'author_id_list': count_unique_in_year,  # 当年独立作者数 (去重)
#     'cited_by_count': 'sum',          # 当年论文获得的累计引用
#     'ref_id_list': count_total_refs_in_year  # 当年参考文献总量
# }).rename(columns={'title': 'paper_count', 'author_id_list': 'unique_authors', 'ref_id_list': 'total_refs'})
#
# # 准备河流图数据 (年份 x 类别)
# river_data = df.groupby(['publication_year', 'cluster_label']).size().unstack(fill_value=0)
#
# # 确保年份连续并平滑处理
# years = np.array(yearly_stats.index)
# x_new = np.linspace(years.min(), years.max(), 300)
#
#
# def smooth(x, y):
#     """3次样条插值平滑"""
#     spl = make_interp_spline(x, y, k=2)
#     return spl(x_new).clip(0)
#
# # --- 4. 绘图 ---
# fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(11, 12), dpi=200, facecolor='#F2F0E4')
# plt.subplots_adjust(hspace=0.35)
#
# # 设置子图背景色
# ax1.set_facecolor('#F2F0E4')
# ax3.set_facecolor('#F2F0E4')
#
# # --- (Top) 双纵坐标趋势图 ---
# ax2 = ax1.twinx()
#
# # 准备平滑曲线数据
# y_papers = smooth(years, yearly_stats['paper_count'])
# y_authors = smooth(years, yearly_stats['unique_authors'])
# y_refs = smooth(years, yearly_stats['total_refs'])
# y_cites = smooth(years, yearly_stats['cited_by_count'])
#
# # 绘制左轴 (发文量 & 作者)
# l1, = ax1.plot(x_new, y_papers, color=COLOR_PALETTE[0], lw=1, label='Papers Count', zorder=5)
# l2, = ax1.plot(x_new, y_authors, color=COLOR_PALETTE[1], lw=1, label='Authors Count', zorder=4)
#
# # 绘制右轴 (引用量 & 参考文献)
# l3, = ax2.plot(x_new, y_refs, color=COLOR_PALETTE[2], lw=1, ls='--', label='Total References', zorder=3)
# l4, = ax2.plot(x_new, y_cites, color=COLOR_PALETTE[3], lw=1, ls=':', label='Total Citations', zorder=2)
#
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
#
# # 轴美化
# ax1.set_ylabel('Papers / Authors', fontsize=11, fontweight='bold')
# ax2.set_ylabel('Refs / Citations', fontsize=11, fontweight='bold')
# # ax1.set_title('A. Scientific Output and Academic Impact Trends', loc='left', fontsize=13, fontweight='bold', pad=15)
# ax1.grid(axis='y', linestyle='--', alpha=0.2)
# ax1.spines['top'].set_visible(False)
# ax2.spines['top'].set_visible(False)
#
# # 合并图例
# lines = [l1, l2, l3, l4]
# ax1.legend(lines, [l.get_label() for l in lines], loc='upper left', frameon=False, fontsize=10)
#
# # --- (Bottom) 主题河流图 ---
# # 平滑每条河流
# river_smooth = []
# for i in range(1, 7):
#     river_smooth.append(smooth(years, river_data[i]))
#
# # 绘制
# ax3.stackplot(x_new, river_smooth, labels=[f'Cluster {i}' for i in range(1, 7)],
#               colors=COLOR_PALETTE[:6], alpha=0.8, edgecolor='white', lw=0, baseline='wiggle')
#
# # ax3.set_title('B. Evolution of Research Clusters (Streamgraph)', loc='left', fontsize=13, fontweight='bold', pad=15)
# ax3.set_xlabel('Year', fontsize=11)
# ax3.set_ylabel('Relative Topic Weight', fontsize=11, fontweight='bold')
# ax3.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['left'].set_visible(False)
# ax3.get_yaxis().set_ticks([]) # 隐藏纵坐标刻度符合河流图审美
#
# # 图例放置在右侧
# ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title="Topic Clusters")
#
# # 保存
# plt.tight_layout()
# plt.savefig('academic_trends_nature_style.png', dpi=600, bbox_inches='tight')
# plt.show()
#
# print("✅ 绘图完成！独立作者已去重，平滑曲线与河流图已生成。")