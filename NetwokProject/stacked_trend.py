import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 配置基础颜色与背景
COLOR_PALETTE = ['#8B7E6F', '#B4C4D5', '#9E9E7E', '#A58B84', '#7E8B9E', '#D6DADB', '#4A453F', '#C2B49B']
BG_COLOR = '#F2F0E4'

# 目标字段及过滤配置
TARGET_COLS = [
    'author_source',
    'overall_interpretability',
    'physics_fusion_depth',
    'robustness_eval',
    'research_level'
]
COLS_TO_FILTER = ['author_source', 'physics_fusion_depth', 'research_level']
EXCLUDE_VALS = {'none', 'nan', 'missing', 'null', '无'}


def generate_stacked_trend_figure(df, year_range=None):
    """
    生成多维度 100% 堆叠面积趋势图 (Plotly 交互版)
    """
    if df is None or df.empty:
        return go.Figure()

    # 1. 过滤年份范围
    filtered_df = df.copy()
    if year_range and 'publication_year' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['publication_year'] >= year_range[0]) & 
            (filtered_df['publication_year'] <= year_range[1])
        ]

    # 创建多子图垂直排列 (5行1列)
    fig = make_subplots(
        rows=len(TARGET_COLS), 
        cols=1,
        subplot_titles=[f"Trend: {col} {'(Filtered None)' if col in COLS_TO_FILTER else ''}" for col in TARGET_COLS],
        vertical_spacing=0.06
    )

    for row_idx, col in enumerate(TARGET_COLS, start=1):
        if col not in filtered_df.columns or 'publication_year' not in filtered_df.columns:
            continue

        temp_df = filtered_df[[col, 'publication_year']].dropna().copy()
        temp_df[col] = temp_df[col].astype(str).str.strip()

        # 执行 None 过滤
        if col in COLS_TO_FILTER:
            temp_df = temp_df[~temp_df[col].str.lower().isin(EXCLUDE_VALS)]

        if temp_df.empty:
            continue

        # 2. 按年份和类别透视分组
        pivot_data = temp_df.groupby(['publication_year', col]).size().unstack(fill_value=0)

        if pivot_data.empty:
            continue

        # 3. 归一化为百分比 (0 - 100%)
        row_sums = pivot_data.sum(axis=1)
        # 防止除以 0
        row_sums[row_sums == 0] = 1
        pivot_perc = pivot_data.divide(row_sums, axis=0) * 100

        years = pivot_perc.index.tolist()
        categories = pivot_perc.columns.tolist()

        # 4. 绘制百分比堆叠面积图 (tonexty 模式)
        for cat_idx, cat_name in enumerate(categories):
            color = COLOR_PALETTE[cat_idx % len(COLOR_PALETTE)]
            
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=pivot_perc[cat_name],
                    name=str(cat_name),
                    legendgroup=f"group_{col}",  # 分组控制图例
                    mode='lines',
                    line=dict(width=0.5, color=color, shape='spline', smoothing=0.8), # spline 平滑线条
                    stackgroup=f'stack_{row_idx}', # 设定同一子图堆叠组
                    groupnorm='percent',           # 自动保持 100% 堆叠
                    fillcolor=color,
                    hovertemplate=f"Year: %{{x}}<br>{col}: {cat_name}<br>Ratio: %{{y:.1f}}%<extra></extra>"
                ),
                row=row_idx, col=1
            )

    # 5. 全局美化与样式整合
    fig.update_layout(
        height=320 * len(TARGET_COLS),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(t=50, b=40, l=50, r=120),
        showlegend=True,
        hovermode="x unified"
    )

    # 统一设置各轴样式
    fig.update_xaxes(showgrid=False, linecolor='#8B7E6F')
    fig.update_yaxes(showgrid=True, gridcolor='#E0DDD0', range=[0, 100], ticksuffix='%')
    
    
    return fig