import pandas as pd
import numpy as np
import plotly.graph_objects as go

COLOR_PALETTE = ['#8B7E6F', '#B4C4D5', '#9E9E7E', '#A58B84']
BG_COLOR = '#F2F0E4'

def generate_citation_histogram(df, year_range=None, color_palette=COLOR_PALETTE):
    """
    绘制引文直方图/分组条形图，展示历年发文量、引用量、被引量、作者数量（去重）。
    """
    data_df = df.copy()

    # 处理作者去重与引用列表提取逻辑
    if 'author_id_list' in data_df.columns:
        data_df['author_id_list'] = data_df['author_id_list'].fillna('').astype(str).apply(
            lambda x: [i.strip() for i in x.split(';') if i.strip()]
        )
    else:
        data_df['author_id_list'] = [[] for _ in range(len(data_df))]

    if 'referenced_ids_openalex' in data_df.columns:
        data_df['ref_id_list'] = data_df['referenced_ids_openalex'].fillna('').astype(str).apply(
            lambda x: [i.strip() for i in x.replace(';', '@').split('@') if i.strip()]
        )
    else:
        data_df['ref_id_list'] = [[] for _ in range(len(data_df))]

    # 根据时间轴进行联动过滤
    if year_range and 'publication_year' in data_df.columns:
        data_df = data_df[(data_df['publication_year'] >= year_range[0]) & 
                          (data_df['publication_year'] <= year_range[1])]

    if data_df.empty or 'publication_year' not in data_df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="当前时间范围内无有效数据",
            paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR
        )
        return fig

    # 按年份统计 4 项指标
    yearly_stats = data_df.groupby('publication_year').agg({
        'title': 'count',
        'author_id_list': lambda s: len(set([item for sublist in s for item in sublist])),
        'cited_by_count': 'sum',
        'ref_id_list': lambda s: len([item for sublist in s for item in sublist])
    }).rename(columns={
        'title': 'paper_count',
        'author_id_list': 'unique_authors',
        'ref_id_list': 'total_refs'
    })

    years = yearly_stats.index.astype(str)

    fig = go.Figure()

    # 1. 发文量
    fig.add_trace(go.Bar(
        x=years,
        y=yearly_stats['paper_count'],
        name='发文量 (Papers)',
        marker_color=color_palette[0]
    ))

    # 2. 作者数量（去重）
    fig.add_trace(go.Bar(
        x=years,
        y=yearly_stats['unique_authors'],
        name='作者数量 (Unique Authors)',
        marker_color=color_palette[1]
    ))

    # 3. 引用量
    fig.add_trace(go.Bar(
        x=years,
        y=yearly_stats['total_refs'],
        name='引用量 (References)',
        marker_color=color_palette[2]
    ))

    # 4. 被引量
    fig.add_trace(go.Bar(
        x=years,
        y=yearly_stats['cited_by_count'],
        name='被引量 (Citations)',
        marker_color=color_palette[3]
    ))

    # 样式与布局设定
    fig.update_layout(
        barmode='group',  # 分组条形图模式
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family='Arial', color='#4A453F'),
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor='rgba(0,0,0,0)'
        )
    )

    fig.update_xaxes(title_text="Publication Year", showgrid=False)
    fig.update_yaxes(title_text="Count / Metrics", showgrid=True, gridcolor='#E0DDD0')

    return fig