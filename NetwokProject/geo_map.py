import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 统一你的全局色盘与背景色
COLOR_LOW = '#C2B49B'   # 低密度米褐色
COLOR_HIGH = '#B84A39'  # 高密度红褐色
BG_COLOR = '#F2F0E4'

def parse_gps(gps_str):
    """ 解析 'lat, lon' 或 '[lat, lon]' 字符串为浮点数 (lat, lon) """
    if pd.isna(gps_str) or not isinstance(gps_str, str):
        return None, None
    clean_str = gps_str.replace('[', '').replace(']', '').replace('(', '').replace(')', '').strip()
    parts = clean_str.split(',')
    if len(parts) == 2:
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            return None, None
    return None, None

def generate_geo_map_figure(df, year_range=None, gps_col='first_affil_city_gps'):
    """
    生成全球论文分布 3D 柱状感/散点地图
    """
    if df is None or df.empty or gps_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR)
        return fig

    # 1. 过滤年份
    filtered_df = df.copy()
    if year_range and 'publication_year' in filtered_df.columns:
        filtered_df['publication_year'] = pd.to_numeric(filtered_df['publication_year'], errors='coerce')
        filtered_df = filtered_df[
            (filtered_df['publication_year'] >= year_range[0]) & 
            (filtered_df['publication_year'] <= year_range[1])
        ]

    # 2. 解析 GPS 经纬度
    gps_data = filtered_df[gps_col].dropna().apply(parse_gps)
    coords = [g for g in gps_data if g[0] is not None and g[1] is not None]

    if not coords:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR)
        return fig

    lat_list, lon_list = zip(*coords)
    coord_df = pd.DataFrame({'lat': lat_list, 'lon': lon_list})

    # 3. 按经纬度聚类汇总数量（聚合近似点，形成类似图片的柱状聚类效果）
    coord_df['lat_round'] = coord_df['lat'].round(1)
    coord_df['lon_round'] = coord_df['lon'].round(1)
    
    geo_counts = coord_df.groupby(['lat_round', 'lon_round']).size().reset_index(name='count')

    # 4. 构建 Plotly Mapbox 图表
    fig = go.Figure()

    # 4.1 底层密度/散点，展现蜂窝散点感
    fig.add_trace(go.Scattermapbox(
        lat=geo_counts['lat_round'],
        lon=geo_counts['lon_round'],
        mode='markers',
        marker=dict(
            size=np.clip(geo_counts['count'] * 2 + 5, 6, 25), # 根据论文数量动态变化尺寸
            color=geo_counts['count'],
            colorscale=[[0, COLOR_LOW], [1, COLOR_HIGH]],
            opacity=0.85,
            showscale=False
        ),
        text=geo_counts['count'],
        hovertemplate="<b>Location:</b> (%{lat}, %{lon})<br><b>Papers Count:</b> %{text}<extra></extra>"
    ))

    # 5. 地图样式与视角配置 (匹配你的淡灰色底图样式)
    fig.update_layout(
        mapbox=dict(
            style="carto-positron", # 极简淡色地图背景，完美契合效果图
            center=dict(lat=20, lon=10),
            zoom=0.8,
            pitch=30 # 倾斜视角，营造 3D 立体视觉感
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=380,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        showlegend=False
    )

    return fig