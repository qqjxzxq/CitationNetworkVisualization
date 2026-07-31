import pandas as pd
import pydeck as pdk

# 主色调配置 (与效果图保持一致的大地暖色系)
BG_COLOR = '#F2F0E4'

def parse_gps(gps_str):
    """ 解析 'lat, lon' 或 '[lat, lon]' 字符串为浮点数 [lon, lat] (PyDeck 要求 [经度, 纬度]) """
    if pd.isna(gps_str) or not isinstance(gps_str, str):
        return None, None
    clean_str = gps_str.replace('[', '').replace(']', '').replace('(', '').replace(')', '').strip()
    parts = clean_str.split(',')
    if len(parts) == 2:
        try:
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            # 过滤无效经纬度
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lon, lat
        except ValueError:
            return None, None
    return None, None

def generate_3d_column_map_html(df, year_range=None, gps_col='first_affil_city_gps'):
    """
    生成真实 3D 蜂窝圆柱体地图 (等比例调低圆柱高度，精致不遮挡视角)
    """
    if df is None or df.empty or gps_col not in df.columns:
        return "<p style='text-align:center;'>No Data Available</p>"

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
        return "<p style='text-align:center;'>No Valid GPS Data</p>"

    coord_df = pd.DataFrame(coords, columns=['lng', 'lat'])

    # 3. 构造 3D Hexagon/Column 柱状图层
    layer = pdk.Layer(
        "HexagonLayer",
        data=coord_df,
        get_position=["lng", "lat"],
        radius=100000,            # 每个圆柱的覆盖半径 (100公里，大小适中)
        elevation_scale=25,      # 【关键修改】大幅降低高度缩放系数 (原500 -> 25)
        elevation_range=[10, 80000], # 【关键修改】限制圆柱的最小与最大绝对高度 (米)
        extruded=True,           # 开启 3D 立体拔高效果
        pickable=True,
        # 渐变颜色 (从低密度的浅米褐色到高密度的深红色)
        color_range=[
            [194, 180, 155, 200],  # #C2B49B (浅米褐色)
            [180, 150, 130, 220],
            [184, 100, 75, 230],
            [184, 74, 57, 240],    # #B84A39 (暖红色)
            [139, 38, 26, 255]     # #8B261A (深红)
        ],
        auto_highlight=True
    )

    # 4. 设置 3D 视角（稍微调小倾斜角 pitch=35，让视角更平稳）
    view_state = pdk.ViewState(
        longitude=15,
        latitude=25,
        zoom=1.2,
        min_zoom=0.5,
        max_zoom=10,
        pitch=35,   # 倾斜 35 度，既能看清高矮差异，又不会遮挡北部地图
        bearing=0
    )

    # 5. 生成 PyDeck 渲染对象
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"html": "<b>Paper Count:</b> {elevationValue}", "style": {"color": "white"}}
    )

    # 导出 HTML 供 Dash 前端 iframe 渲染
    return deck.to_html(as_string=True)