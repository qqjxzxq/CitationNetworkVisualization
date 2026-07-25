import io
import base64
from collections import Counter
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud

# 全局样式与莫兰迪配色方案
COLOR_PALETTE = ['#8B7E6F', '#B4C4D5', '#9E9E7E', '#A58B84', '#7E8B9E', '#D6DADB', '#4A453F', '#C2B49B']
BG_COLOR = '#F2F0E4'


def get_frequencies(series):
    """提取与统计 concepts 词频"""
    all_concepts = []
    for item in series.dropna():
        all_concepts.extend([x.strip() for x in str(item).split(';') if x.strip()])
    return Counter(all_concepts)


def generate_wordcloud_base64(counts, width=800, height=500):
    """生成词云图并转换为 Base64 格式字符串"""
    if not counts:
        return ""

    wc = WordCloud(
        width=width,
        height=height,
        background_color=BG_COLOR,
        prefer_horizontal=0.7,  # 70%横向，30%纵向成直角
        color_func=lambda *args, **kwargs: COLOR_PALETTE[hash(args[0]) % len(COLOR_PALETTE)], # 保持色彩均匀分布
        max_words=100
    ).generate_from_frequencies(counts)

    # 保存到内存二进制流并转 Base64
    img_bytes = io.BytesIO()
    wc.to_image().save(img_bytes, format='PNG')
    return "data:image/png;base64," + base64.b64encode(img_bytes.getvalue()).decode('utf-8')


def generate_top10_barchart(counts):
    """生成右侧的 Top 10 Plotly 柱状图"""
    if not counts:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR)
        return fig

    # 提取 Top 10 数据（并反转顺序，使最高频排在上方）
    top_10 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10.reverse()
    words, values = zip(*top_10)

    fig = go.Figure(go.Bar(
        x=values,
        y=words,
        orientation='h',
        marker=dict(
            color=[COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(words))],
            line=dict(width=0)
        )
    ))

    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        xaxis=dict(showgrid=True, gridcolor='#E0DCD3', zeroline=False, title=''),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color='#4A453F')),
        height=320
    )
    return fig


def get_wordcloud_and_bar_assets(df_filtered, target_col='concepts'):
    """
    对外调用的主接口函数
    接收过滤后的 DataFrame，返回 (词云Base64图片, Plotly条形图Figure)
    """
    if df_filtered.empty or target_col not in df_filtered.columns:
        return "", generate_top10_barchart(Counter())

    counts = get_frequencies(df_filtered[target_col])
    img_base64 = generate_wordcloud_base64(counts)
    bar_fig = generate_top10_barchart(counts)

    return img_base64, bar_fig