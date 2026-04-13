import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from datashader.bundling import hammer_bundle


# --- 1. 数据预处理（计算固定坐标） ---
def load_and_layout():
    # 加载数据
    df_sample = pd.read_csv('sample.csv')
    df_umap = pd.read_csv('abstract_umap.csv').drop_duplicates(subset=['magid'])
    df_umap['magid'] = df_umap['magid'].astype(str)

    # 元数据合并
    meta = df_sample[['paper_openalex_id', 'title', 'publication_year', 'abstract', 'cited_by_count',
                      'referenced_ids_openalex', 'author_id_list', 'author_list']].drop_duplicates(subset=['paper_openalex_id'])
    meta['paper_openalex_id'] = meta['paper_openalex_id'].astype(str)

    nodes_data = df_umap.set_index('magid')
    nodes_data = nodes_data.join(meta.set_index('paper_openalex_id')).fillna({
        'cited_by_count': 0, 'title': 'Unknown', 'abstract': 'No abstract available.', 'publication_year': 2000
    })

    # 1.1 节点基础布局计算 (固定坐标部分)
    N_CLUSTERS =6
    WEIGHT_CLUSTER, WEIGHT_CITATION, ITERATIONS = 0.7, 0.3, 60

    raw_thetas = np.arctan2(nodes_data['ys'], nodes_data['xs'])
    nodes_data['theta_init'] = raw_thetas.rank(pct=True) * 2 * np.pi
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    nodes_data['cluster'] = kmeans.fit_predict(nodes_data[['xs', 'ys']])
    cluster_centers = nodes_data.groupby('cluster')['theta_init'].median().to_dict()

    min_yr, max_yr = nodes_data['publication_year'].min(), nodes_data['publication_year'].max()
    nodes_data['r'] = 0.2 + 0.8 * (nodes_data['publication_year'] - min_yr) / (max_yr - min_yr + 1e-5)

    # 提取所有边
    all_edges = []
    for _, row in meta.iterrows():
        sid = str(row['paper_openalex_id'])
        if pd.notna(row['referenced_ids_openalex']):
            targets = str(row['referenced_ids_openalex']).replace(';', '@').split('@')
            for tid in [t.strip() for t in targets]:
                if sid in nodes_data.index and tid in nodes_data.index:
                    all_edges.append((sid, tid))

    # 坐标迭代
    pos = {n: np.array([nodes_data.at[n, 'r'] * np.cos(nodes_data.at[n, 'theta_init']),
                        nodes_data.at[n, 'r'] * np.sin(nodes_data.at[n, 'theta_init'])]) for n in nodes_data.index}

    for _ in range(ITERATIONS):
        new_pos = pos.copy()
        for u, v in all_edges:
            delta = pos[v] - pos[u]
            dist = np.linalg.norm(delta) + 1e-6
            force = dist * 0.08 * WEIGHT_CITATION
            new_pos[u] = new_pos[u] + (delta / dist) * force
            new_pos[v] = new_pos[v] - (delta / dist) * force
        for n in nodes_data.index:
            curr_theta = np.arctan2(new_pos[n][1], new_pos[n][0])
            c_theta = cluster_centers[nodes_data.at[n, 'cluster']]
            s_theta = nodes_data.at[n, 'theta_init']
            semantic_base = s_theta + WEIGHT_CLUSTER * np.arctan2(np.sin(c_theta - s_theta), np.cos(c_theta - s_theta))
            final_theta = semantic_base + WEIGHT_CITATION * np.arctan2(np.sin(curr_theta - semantic_base),
                                                                       np.cos(curr_theta - semantic_base))
            pos[n] = np.array(
                [nodes_data.at[n, 'r'] * np.cos(final_theta), nodes_data.at[n, 'r'] * np.sin(final_theta)])

    nodes_data['x'] = [pos[n][0] for n in nodes_data.index]
    nodes_data['y'] = [pos[n][1] for n in nodes_data.index]
    
    # 1.2 解析作者数据
    author_data = {} 
    author_collab = {} 

    for _, row in meta.iterrows():
        pid = str(row['paper_openalex_id'])
        if pid not in nodes_data.index: continue
        
        # 假设你的列名是 author_id_list 和 author_list
        if pd.isna(row['author_id_list']): continue
        ids = str(row['author_id_list']).split(';')
        names = str(row['author_list']).split(';')
        
        p_x, p_y = nodes_data.at[pid, 'x'], nodes_data.at[pid, 'y']
        p_year, p_cite = row['publication_year'], row['cited_by_count']

        for i, aid in enumerate(ids):
            aid = aid.strip()
            if not aid: continue
            if aid not in author_data:
                author_data[aid] = {'name': names[i].strip(), 'years': [], 'xs': [], 'ys': [], 'cites': 0}
            author_data[aid]['xs'].append(p_x)
            author_data[aid]['ys'].append(p_y)
            author_data[aid]['years'].append(p_year)
            author_data[aid]['cites'] += p_cite

        # 建立协作边
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair = tuple(sorted([ids[i].strip(), ids[j].strip()]))
                if pair[0] and pair[1]:
                    author_collab[pair] = author_collab.get(pair, 0) + 1

    # 1.3 构建作者节点 DataFrame
    author_rows = []
    for aid, info in author_data.items():
        author_rows.append({
            'author_id': aid,
            'name': info['name'],
            'x': np.mean(info['xs']),  # 作者位置是其论文位置的平均值
            'y': np.mean(info['ys']),
            'publication_year': np.min(info['years']), # 首次活跃年份
            'cited_by_count': info['cites'],
            'cluster': 0 # 也可以通过 KMeans 重新聚类，此处简写
        })
    nodes_author = pd.DataFrame(author_rows).set_index('author_id')
    # 简单的聚类分配
    kmeans_a = KMeans(n_clusters=6, random_state=42, n_init=10)
    nodes_author['cluster'] = kmeans_a.fit_predict(nodes_author[['x', 'y']])

    # 1.4 提取作者边
    edges_author = [list(p) for p in author_collab.keys() if p[0] in nodes_author.index and p[1] in nodes_author.index]

    return nodes_data, all_edges, nodes_author, edges_author, int(min_yr), int(max_yr)

# 初始化数据
nodes_df, edges_pool, nodes_author, edges_author, MIN_Y, MAX_Y = load_and_layout()

# --- 2. Dash 网页布局 ---
app = Dash(__name__)

COLOR_PALETTE = ['#8B7E6F', '#B4C4D5', '#9E9E7E', '#A58B84', '#7E8B9E', '#D6DADB', '#4A453F', '#C2B49B']

app.layout = html.Div(style={'backgroundColor': '#F2F0E4', 'minHeight': '100vh', 'padding': '20px'}, children=[
    html.H2("文献引文网络 - 交互式可视化", style={'textAlign': 'center', 'color': '#4A453F', 'marginBottom': '20px'}),

    # 控制面板
    html.Div([
        html.Label("🌐 视图模式:", style={'fontWeight': 'bold', 'color': '#4A453F'}),
        dcc.RadioItems(
            id='view-mode',
            options=[
                {'label': ' 论文引文网络', 'value': 'paper'},
                {'label': ' 作者协作网络', 'value': 'author'}
            ],
            value='paper', # 默认显示论文
            labelStyle={'display': 'inline-block', 'marginRight': '20px', 'marginTop': '5px'}
        )
    ], style={'marginBottom': '20px', 'paddingBottom': '15px', 'borderBottom': '1px solid #eee'}),
    
    html.Div([
        # 第一排：搜索和年份
        html.Div([
            html.Div([
                html.Label("🔍 模糊搜索:", style={'fontWeight': 'bold'}),
                dcc.Input(id='search-box', type='text', placeholder='标题关键词...',
                          style={'width': '90%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
            ], style={'width': '33%', 'display': 'inline-block'}),
            html.Div([
                html.Label("📅 出版年份范围:", style={'fontWeight': 'bold'}),
                dcc.RangeSlider(id='year-slider', min=MIN_Y, max=MAX_Y, step=1, value=[MIN_Y, MAX_Y],
                                marks={i: str(i) for i in range(MIN_Y, MAX_Y + 1, 5)})
            ], style={'width': '66%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'marginBottom': '20px'}),

        # 第二排：节点大小控制
        html.Div([
            html.Div([
                html.Label("🔘 基础节点大小 (Base Size):", style={'fontWeight': 'bold'}),
                dcc.Slider(id='base-size-slider', min=1, max=20, step=0.5, value=5,
                           tooltip={"placement": "bottom", "always_visible": True})
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                html.Label("🚀 引用缩放增量 (Scaling Factor):", style={'fontWeight': 'bold'}),
                dcc.Slider(id='scale-factor-slider', min=0, max=100, step=5, value=35,
                           tooltip={"placement": "bottom", "always_visible": True})
            ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
        ])
    ], style={'background': 'white', 'padding': '20px', 'borderRadius': '10px',
              'boxShadow': '0 2px 10px rgba(0,0,0,0.05)', 'marginBottom': '20px'}),

    # 主绘图区
    html.Div([
        dcc.Graph(id='main-plot', config={'displayModeBar': False},
                  style={'height': '80vh', 'width': '80vh', 'margin': '0 auto'}),
        # 信息详情面板
        html.Div(id='info-panel', style={
            'position': 'absolute', 'top': '20px', 'right': '20px', 'width': '320px',
            'backgroundColor': 'rgba(255, 255, 255, 0.95)', 'padding': '20px',
            'borderRadius': '8px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.15)',
            'display': 'none', 'maxHeight': '80%', 'overflowY': 'auto', 'border': '1px solid #8B7E6F', 'zIndex': '1000'
        })
    ], style={'position': 'relative', 'textAlign': 'center'})
])


# --- 3. 交互逻辑回调 ---
@app.callback(
    Output('main-plot', 'figure'),
    [Input('view-mode', 'value'),
     Input('year-slider', 'value'),
     Input('search-box', 'value'),
     Input('base-size-slider', 'value'),
     Input('scale-factor-slider', 'value')]
)
def update_network(view_mode, years, search_txt, base_size, scale_factor):
    # 0. 数据源切换逻辑 
    if view_mode == 'paper':
        df = nodes_df  # 原始论文数据
        edges_pool_to_use = edges_pool
        label_col = 'title'
        # hover_extra = 'abstract'
    else:
        df = nodes_author  # 新增的作者数据
        edges_pool_to_use = edges_author
        label_col = 'name'
        # hover_extra = 'name' # 作者没有摘要，重复显示名字或留空
        
    # 1. 过滤节点
    filtered_nodes = df[(df['publication_year'] >= years[0]) & (df['publication_year'] <= years[1])].copy()
    node_ids = set(filtered_nodes.index)

    # 2. 动态计算节点大小
    sqrt_cites = np.sqrt(filtered_nodes['cited_by_count'])
    # 计算公式应用前端传来的 base_size 和 scale_factor
    filtered_nodes['node_s'] = base_size + (sqrt_cites / (sqrt_cites.max() + 1)) * scale_factor

    # 3. 边捆绑计算
    current_edges = [(u, v) for u, v in edges_pool_to_use if u in node_ids and v in node_ids]
    edge_x, edge_y = [], []
    if current_edges:
        nodes_for_hb = filtered_nodes[['x', 'y']]
        edges_for_hb = pd.DataFrame(current_edges, columns=['source', 'target'])
        hb_paths = hammer_bundle(nodes_for_hb, edges_for_hb, initial_bandwidth=0.1, decay=0.7)
        edge_x = hb_paths['x'].tolist()
        edge_y = hb_paths['y'].tolist()

    # 4. 搜索高亮
    marker_line_widths = [0] * len(filtered_nodes)
    if search_txt and len(search_txt) > 1:
        highlight_idx = filtered_nodes[label_col].str.contains(search_txt, case=False, na=False)
        marker_line_widths = [2.5 if val else 0 for val in highlight_idx]

    # 5. 构建图形
    fig = go.Figure()

    # 背景环
    for y_val in np.linspace(MIN_Y, MAX_Y, 6):
        r_val = 0.2 + 0.8 * (y_val - MIN_Y) / (MAX_Y - MIN_Y + 1e-5)
        fig.add_shape(type="circle", xref="x", yref="y", x0=-r_val, y0=-r_val, x1=r_val, y1=r_val,
                      line=dict(color="#D6DADB", width=1, dash="dot"))

    # 边
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.6, color='#4A453F'),
                             hoverinfo='none', mode='lines', opacity=0.2))

    # 节点
    fig.add_trace(go.Scatter(
        x=filtered_nodes['x'], y=filtered_nodes['y'],
        mode='markers',
        text=filtered_nodes[label_col], # 动态标签
        # 作者模式下处理 customdata
        customdata=np.stack((
            filtered_nodes['abstract'] if view_mode == 'paper' else filtered_nodes['name'], 
            filtered_nodes.index
        ), axis=-1),
        marker=dict(
            size=filtered_nodes['node_s'],
            color=[COLOR_PALETTE[c % 8] for c in filtered_nodes['cluster']],
            line=dict(width=marker_line_widths, color='red'),
            opacity=0.8
        ),
        hoverinfo='text'
    ))

    fig.update_layout(
        showlegend=False, clickmode='event',
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='#F2F0E4', plot_bgcolor='#F2F0E4',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2], fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2],
                   scaleanchor="x", scaleratio=1, fixedrange=True)
    )
    return fig


# 4. 处理点击与取消逻辑
@app.callback(
    [Output('info-panel', 'children'), Output('info-panel', 'style')],
    [Input('main-plot', 'clickData')],
    prevent_initial_call=False
)
def handle_click(clickData):
    if not clickData or 'points' not in clickData or 'customdata' not in clickData['points'][0]:
        return "", {'display': 'none'}

    point = clickData['points'][0]
    title = point['text']
    info = point['customdata'][0]

    is_author_mode = (title == info)
    label = "个人简介/姓名: " if is_author_mode else "摘要: "
    display_text = info if not is_author_mode else f"选定作者：{info}"
    panel_content = html.Div([
        html.H3(title, style={'color': '#4A453F', 'fontSize': '16px', 'borderBottom': '1px solid #ccc',
                              'paddingBottom': '10px'}),
        html.P([html.Strong(label), display_text],
               style={'fontSize': '13px', 'lineHeight': '1.5', 'textAlign': 'justify'}),
        html.Hr(),
        html.Em("提示: 点击图表空白区域可关闭此面板", style={'fontSize': '11px', 'color': '#999'})
    ])

    panel_style = {
        'position': 'absolute', 'top': '20px', 'right': '20px', 'width': '320px',
        'backgroundColor': 'rgba(255, 255, 255, 0.98)', 'padding': '20px',
        'borderRadius': '12px', 'boxShadow': '0 8px 30px rgba(0,0,0,0.2)',
        'display': 'block', 'maxHeight': '80%', 'overflowY': 'auto',
        'border': '1px solid #8B7E6F', 'zIndex': '1000'
    }
    return panel_content, panel_style


if __name__ == '__main__':
    app.run(debug=True)