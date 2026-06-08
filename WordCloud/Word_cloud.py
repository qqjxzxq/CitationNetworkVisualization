import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import matplotlib.colors as mcolors

# 加载数据
df = pd.read_csv('combined_data_updated.csv')
COLOR_PALETTE = ['#8B7E6F', '#B4C4D5', '#9E9E7E', '#A58B84', '#7E8B9E', '#D6DADB', '#4A453F', '#C2B49B']
BG_COLOR = '#F2F0E4'


def get_frequencies(series):
    # 处理分号分隔的 concept
    all_concepts = []
    for item in series.dropna():
        all_concepts.extend([x.strip() for x in str(item).split(';') if x.strip()])
    return Counter(all_concepts)


def plot_set(counts, title):
    # 莫兰蒂色系
    cmap = mcolors.ListedColormap(COLOR_PALETTE)

    # 创建画布，设置背景色
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), facecolor=BG_COLOR)

    # 1. 词云图部分
    wc = WordCloud(width=800, height=400,
                   background_color=BG_COLOR,
                   colormap=cmap,
                   prefer_horizontal=0.7).generate_from_frequencies(counts)
    ax1.imshow(wc, interpolation='bilinear')
    ax1.axis('off')  # 词云图自带去边框
    ax1.set_title(f"{title} Word Cloud", fontsize=16, pad=20, color='#4A453F', fontweight='bold')

    # 2. 直方图部分 (Top 10)
    top_10 = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10])
    labels = list(top_10.keys())[::-1]
    values = list(top_10.values())[::-1]

    bars = ax2.barh(labels, values, color=COLOR_PALETTE)
    ax2.set_facecolor(BG_COLOR)

    # --- 核心：去掉边框和刻度 ---
    for spine in ['top', 'right', 'bottom', 'left']:
        ax2.spines[spine].set_visible(False)

    ax2.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
    ax2.set_title(f"{title} Top 10 Concepts", fontsize=14, color='#4A453F')

    plt.tight_layout(pad=4.0)
    plt.show()


# 运行全集
plot_set(get_frequencies(df['concepts']), "Overall Dataset")

# 运行各 Cluster
for i in range(1, 7):
    cluster_df = df[df['cluster_label'] == i]
    if not cluster_df.empty:
        plot_set(get_frequencies(cluster_df['concepts']), f"Cluster {i}")