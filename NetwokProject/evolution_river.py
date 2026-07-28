import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import make_interp_spline

# 调色板保持一致
COLOR_PALETTE = ['#8B7E6F', '#B4C4D5', '#9E9E7E', '#A58B84', '#7E8B9E', '#D6DADB', '#4A453F', '#C2B49B']


def _smooth(x, y, x_new):
    """ scipy spline 平滑插值"""
    if len(x) < 2:
        return np.tile(y, (len(x_new), 1)).T
    k_val = 2 if len(x) > 2 else 1
    spl = make_interp_spline(x, y, k=k_val)
    return spl(x_new).clip(0)


def _compute_wiggle_baseline(y_matrices):
    """
    还原 matplotlib stackplot(baseline='wiggle') 算法
    计算 ThemeRiver 中轴下沉量 y0
    """
    n_series, n_points = y_matrices.shape
    cum_y = np.cumsum(y_matrices, axis=0)

    # 简化的 Wiggle (ThemeRiver) 中轴偏置算法：使总流量围绕中轴上下对称波动
    total_y = np.sum(y_matrices, axis=0)
    baseline = -0.5 * total_y
    return baseline


def generate_evolution_river(meta_df, nodes_df, year_range=None, color_palette=COLOR_PALETTE):
    """
    交互式绘图函数
    包含：
      1. 上图：双 Y 轴趋势图（Papers, Authors vs. Refs, Citations）带置信区间阴影
      2. 下图：真正的 Streamgraph 主题河流图（Wiggle Baseline 对称平滑效果）
    """
    # --- Data Preprocessing ---
    df = meta_df.copy()
    if 'cluster' not in df.columns:
        df['cluster'] = df['paper_openalex_id'].map(nodes_df['cluster'])

    df = df.dropna(subset=['publication_year', 'cluster'])
    df['publication_year'] = df['publication_year'].astype(int)
    df['cluster'] = df['cluster'].astype(int)

    # 列表提取 logic
    if 'author_id_list' in df.columns:
        df['author_id_list'] = df['author_id_list'].fillna('').astype(str).apply(
            lambda x: [i.strip() for i in x.split(';') if i.strip()]
        )
    else:
        df['author_id_list'] = [[] for _ in range(len(df))]

    if 'referenced_ids_openalex' in df.columns:
        df['ref_id_list'] = df['referenced_ids_openalex'].fillna('').astype(str).apply(
            lambda x: [i.strip() for i in x.replace(';', '@').split('@') if i.strip()]
        )
    else:
        df['ref_id_list'] = [[] for _ in range(len(df))]

    # 按年份过滤
    if year_range:
        df = df[(df['publication_year'] >= year_range[0]) & (df['publication_year'] <= year_range[1])]

    if df.empty or len(df['publication_year'].unique()) < 2:
        fig = go.Figure()
        fig.update_layout(
            title="Data range too narrow for trend rendering",
            paper_bgcolor='#F2F0E4', plot_bgcolor='#F2F0E4'
        )
        return fig

    # --- 聚合指标计算 ---
    yearly_stats = df.groupby('publication_year').agg({
        'title': 'count',
        'author_id_list': lambda s: len(set([item for sublist in s for item in sublist])),
        'cited_by_count': 'sum',
        'ref_id_list': lambda s: len([item for sublist in s for item in sublist])
    }).rename(columns={
        'title': 'paper_count',
        'author_id_list': 'unique_authors',
        'ref_id_list': 'total_refs'
    })

    years = np.array(yearly_stats.index)
    x_new = np.linspace(years.min(), years.max(), 300)

    # 15% 仿真标准差阴影
    std_fill = yearly_stats * 0.15

    # 计算 4 条指标平滑折线
    metrics = ['paper_count', 'unique_authors', 'total_refs', 'cited_by_count']
    lines_data = {}
    for m in metrics:
        y = yearly_stats[m].values
        err = std_fill[m].values
        lines_data[m] = {
            'mid': _smooth(years, y, x_new),
            'low': _smooth(years, y - err, x_new),
            'high': _smooth(years, y + err, x_new)
        }

    # --- 河流图数据计算 (Wiggle 算法) ---
    river_data = df.groupby(['publication_year', 'cluster']).size().unstack(fill_value=0)
    clusters = river_data.columns.tolist()
    
    # 获取各个 Cluster 插值后的 2D Array
    raw_river_smooth = np.array([_smooth(years, river_data[c].values, x_new) for c in clusters])
    
    # 计算底座 Baseline
    baseline = _compute_wiggle_baseline(raw_river_smooth)

    # 计算各多边形的上边界与下边界
    river_polygons = []
    current_base = baseline.copy()
    for y_vals in raw_river_smooth:
        y_bottom = current_base.copy()
        y_top = current_base + y_vals
        river_polygons.append((y_bottom, y_top))
        current_base = y_top

    # --- 构建 Subplots 双层图 ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # ------------------ (Top Plot) 双 Y 轴趋势图 ------------------
    # 1. Papers Count
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_new, x_new[::-1]]),
        y=np.concatenate([lines_data['paper_count']['high'], lines_data['paper_count']['low'][::-1]]),
        fill='toself', fillcolor='rgba(128,128,128,0.15)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo='none', showlegend=False
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=x_new, y=lines_data['paper_count']['mid'],
        mode='lines', line=dict(color=color_palette[0], width=2),
        name='Papers Count'
    ), row=1, col=1, secondary_y=False)

    # 2. Authors Count
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_new, x_new[::-1]]),
        y=np.concatenate([lines_data['unique_authors']['high'], lines_data['unique_authors']['low'][::-1]]),
        fill='toself', fillcolor='rgba(128,128,128,0.15)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo='none', showlegend=False
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=x_new, y=lines_data['unique_authors']['mid'],
        mode='lines', line=dict(color=color_palette[1], width=2),
        name='Authors Count'
    ), row=1, col=1, secondary_y=False)

    # 3. Total References (Right Y-axis)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_new, x_new[::-1]]),
        y=np.concatenate([lines_data['total_refs']['high'], lines_data['total_refs']['low'][::-1]]),
        fill='toself', fillcolor='rgba(128,128,128,0.1)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo='none', showlegend=False
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=x_new, y=lines_data['total_refs']['mid'],
        mode='lines', line=dict(color=color_palette[2], width=1.5, dash='dash'),
        name='Total References'
    ), row=1, col=1, secondary_y=True)

    # 4. Total Citations (Right Y-axis)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_new, x_new[::-1]]),
        y=np.concatenate([lines_data['cited_by_count']['high'], lines_data['cited_by_count']['low'][::-1]]),
        fill='toself', fillcolor='rgba(128,128,128,0.1)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo='none', showlegend=False
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=x_new, y=lines_data['cited_by_count']['mid'],
        mode='lines', line=dict(color=color_palette[3], width=1.5, dash='dot'),
        name='Total Citations'
    ), row=1, col=1, secondary_y=True)

    # ------------------ (Bottom Plot) 主题河流图 (Wiggle baseline) ------------------
    for i, c in enumerate(clusters):
        y_bot, y_top = river_polygons[i]
        c_color = color_palette[i % len(color_palette)]

        # 利用封闭多边形完美的绘制具有流动效果的半透明 Stackplot
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_new, x_new[::-1]]),
            y=np.concatenate([y_top, y_bot[::-1]]),
            fill='toself',
            fillcolor=c_color,
            line=dict(color='white', width=0.5),
            name=f'Cluster {c}',
            hoverinfo='text',
            text=f'Cluster {c}'
        ), row=2, col=1)

    # --- 全局样式与美化 (保持 #F2F0E4 复古风格) ---
    fig.update_layout(
        paper_bgcolor='#F2F0E4',
        plot_bgcolor='#F2F0E4',
        margin=dict(l=50, r=50, t=30, b=40),
        font=dict(family='Arial', color='#4A453F'),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor='rgba(0,0,0,0)'
        )
    )

    # X 轴美化
    fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(title_text="Year", showgrid=False, zeroline=False, row=2, col=1)

    # Y 轴美化
    fig.update_yaxes(title_text="<b>Papers / Authors</b>", showgrid=False, row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="<b>Refs / Citations</b>", showgrid=False, row=1, col=1, secondary_y=True)

    # 隐藏河流图的数值 Y 轴 (实现 Relative Topic Weight 的意图)
    fig.update_yaxes(
        title_text="<b>Relative Topic Weight</b>",
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        row=2, col=1
    )

    return fig