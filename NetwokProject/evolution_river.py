import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 调色板定义
COLOR_PALETTE = ['#8B7E6F', '#B4C4D5', '#9E9E7E', '#A58B84', '#7E8B9E', '#D6DADB', '#4A453F', '#C2B49B']


def generate_evolution_river(meta_df, nodes_df, year_range=None, color_palette=COLOR_PALETTE):
    """根据论文元数据与节点聚类数据，生成 Plotly 交互式的知识演化趋势图与河流图。

    参数:
        meta_df (pd.DataFrame): 包含 publication_year, author_id_list, cited_by_count, referenced_ids_openalex
          的数据
        nodes_df (pd.DataFrame): 包含 cluster 信息的节点数据表
        year_range (tuple/list, optional): [start_year, end_year] 年份筛选范围
        color_palette (list): 颜色主题列表

    返回:
        fig (plotly.graph_objects.Figure): 可直接给 Dash dcc.Graph 使用的 Plotly 图表对象
    """
    # --- 1. 数据预处理 ---
    df = meta_df.copy()
    if 'cluster' not in df.columns:
        df['cluster'] = df['paper_openalex_id'].map(nodes_df['cluster'])

    # 过滤无效数据与年份区间
    df = df.dropna(subset=['publication_year', 'cluster'])
    df['publication_year'] = df['publication_year'].astype(int)
    df['cluster'] = df['cluster'].astype(int)

    if year_range:
        df = df[(df['publication_year'] >= year_range[0]) & (df['publication_year'] <= year_range[1])]

    # 空数据安全防护
    if df.empty or len(df['publication_year'].unique()) < 2:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Selected year range contains insufficient data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#8B7E6F"),
        )
        empty_fig.update_layout(paper_bgcolor='#F2F0E4', plot_bgcolor='#F2F0E4')
        return empty_fig

    # 清理拆分列表数据
    df['author_id_list'] = (
        df['author_id_list']
        .fillna('')
        .astype(str)
        .apply(lambda x: [i.strip() for i in x.split(';') if i.strip()])
    )
    df['ref_id_list'] = (
        df['referenced_ids_openalex']
        .fillna('')
        .astype(str)
        .apply(lambda x: [i.strip() for i in x.split('@') if i.strip()])
    )

    # --- 2. 统计计算 ---
    def count_unique(series):
        return len(set(item for sublist in series for item in sublist))

    def count_total(series):
        return len([item for sublist in series for item in sublist])

    yearly_stats = (
        df.groupby('publication_year')
        .agg({
            'title': 'count',
            'author_id_list': count_unique,
            'cited_by_count': 'sum',
            'ref_id_list': count_total,
        })
        .rename(
            columns={
                'title': 'paper_count',
                'author_id_list': 'unique_authors',
                'ref_id_list': 'total_refs',
            }
        )
    )

    std_fill = yearly_stats * 0.15
    years = np.array(yearly_stats.index)

    # 平滑插值计算
    if len(years) < 3:
        x_new = years

        def smooth(x, y):
            return y

    else:
        x_new = np.linspace(years.min(), years.max(), 200)

        def smooth(x, y):
            spl = make_interp_spline(x, y, k=min(2, len(years) - 1))
            return spl(x_new).clip(0)

    # --- 3. 构建 Plotly 子图结构 ---
    # 创建上下两个 Row 子图，并为 Top 视图配置双 Y 轴 (secondary_y=True)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=("📈 Publishing & Citation Trends", "🌊 Topic Knowledge Evolution Stream"),
    )

    # --- (Top) 趋势指标图数据添加 ---
    metrics = ['paper_count', 'unique_authors', 'total_refs', 'cited_by_count']
    lines_data = {}
    for m in metrics:
        y = yearly_stats[m]
        error = std_fill[m]
        lines_data[m] = {
            'mid': smooth(years, y),
            'low': smooth(years, y - error),
            'high': smooth(years, y + error),
        }

    # 1. Papers Count (左 Y 轴)
    fig.add_trace(
        go.Scatter(
            x=x_new,
            y=lines_data['paper_count']['mid'],
            name='Papers Count',
            line=dict(color=color_palette[0], width=2.5),
            mode='lines',
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # 2. Authors Count (左 Y 轴)
    fig.add_trace(
        go.Scatter(
            x=x_new,
            y=lines_data['unique_authors']['mid'],
            name='Authors Count',
            line=dict(color=color_palette[1], width=2.5),
            mode='lines',
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # 3. Total References (右 Y 轴)
    fig.add_trace(
        go.Scatter(
            x=x_new,
            y=lines_data['total_refs']['mid'],
            name='Total References',
            line=dict(color=color_palette[2], width=2, dash='dash'),
            mode='lines',
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # 4. Total Citations (右 Y 轴)
    fig.add_trace(
        go.Scatter(
            x=x_new,
            y=lines_data['cited_by_count']['mid'],
            name='Total Citations',
            line=dict(color=color_palette[3], width=2, dash='dot'),
            mode='lines',
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # --- (Bottom) 主题河流图数据添加 ---
    river_data = df.groupby(['publication_year', 'cluster']).size().unstack(fill_value=0)
    clusters = river_data.columns

    # 堆叠图 (Stacked Area) 转换
    for idx, col in enumerate(clusters):
        y_smooth = smooth(years, river_data[col])
        cluster_color = color_palette[int(col) % len(color_palette)]

        fig.add_trace(
            go.Scatter(
                x=x_new,
                y=y_smooth,
                name=f'Cluster {col}',
                mode='lines',
                line=dict(width=0.5, color=cluster_color),
                stackgroup='one',  # 启用自动堆叠实现河流/面积流效果
                fillcolor=cluster_color,
                groupnorm='',  # 也可替换为 'fraction' 来观察相对比例
                hoverinfo='x+y+name',
            ),
            row=2,
            col=1,
        )

    # --- 4. 统一画布美化与样式升级 ---
    fig.update_layout(
        height=650,
        paper_bgcolor='#F2F0E4',
        plot_bgcolor='#F2F0E4',
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.6)',
        ),
    )

    # 坐标轴美化
    fig.update_xaxes(showgrid=True, gridcolor='#E5E3D7', row=1, col=1)
    fig.update_xaxes(title_text="Publication Year", showgrid=True, gridcolor='#E5E3D7', row=2, col=1)

    fig.update_yaxes(
        title_text="Papers / Authors",
        showgrid=True,
        gridcolor='#E5E3D7',
        secondary_y=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Refs / Citations", showgrid=False, secondary_y=True, row=1, col=1
    )
    fig.update_yaxes(
        title_text="Topic Volume", showgrid=True, gridcolor='#E5E3D7', row=2, col=1
    )

    return fig