from collections import Counter
import json
import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, clientside_callback
from datashader.bundling import hammer_bundle
import llm_helper
from wordcloud_module import get_wordcloud_and_bar_assets
import evolution_river  
import stacked_trend
import citation_histogram
import geo_map_3d

# --- 1. Data Preprocessing ---
def load_and_layout():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(current_dir, 'sample.csv')
    umap_path = os.path.join(current_dir, 'abstract_umap.csv')
    
    global stacked_data_df
    stacked_data_df = pd.read_csv('data_for_stacked.csv')

    df_sample = pd.read_csv(sample_path)
    df_umap = pd.read_csv(umap_path).drop_duplicates(subset=['magid'])
    df_umap['magid'] = df_umap['magid'].astype(str)

    # Metadata Merging
    meta = df_sample[['paper_openalex_id', 'title', 'publication_year', 'abstract', 'cited_by_count',
                      'referenced_ids_openalex', 'author_id_list', 'author_list', 'concepts']].drop_duplicates(subset=['paper_openalex_id'])
    meta['paper_openalex_id'] = meta['paper_openalex_id'].astype(str)

    nodes_data = df_umap.set_index('magid')
    nodes_data = nodes_data.join(meta.set_index('paper_openalex_id')).fillna({
        'cited_by_count': 0, 'title': 'Unknown', 'abstract': 'No abstract available.', 'publication_year': 2000
    })

    # 1.1 Base Node Layout Calculation
    N_CLUSTERS = 6
    WEIGHT_CLUSTER, WEIGHT_CITATION, ITERATIONS = 0.7, 0.3, 60

    raw_thetas = np.arctan2(nodes_data['ys'], nodes_data['xs'])
    nodes_data['theta_init'] = raw_thetas.rank(pct=True) * 2 * np.pi
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    nodes_data['cluster'] = kmeans.fit_predict(nodes_data[['xs', 'ys']])
    cluster_centers = nodes_data.groupby('cluster')['theta_init'].median().to_dict()

    min_yr, max_yr = nodes_data['publication_year'].min(), nodes_data['publication_year'].max()
    nodes_data['r'] = 0.2 + 0.8 * (nodes_data['publication_year'] - min_yr) / (max_yr - min_yr + 1e-5)

    all_edges = []
    for _, row in meta.iterrows():
        sid = str(row['paper_openalex_id'])
        if pd.notna(row['referenced_ids_openalex']):
            targets = str(row['referenced_ids_openalex']).replace(';', '@').split('@')
            for tid in [t.strip() for t in targets]:
                if sid in nodes_data.index and tid in nodes_data.index:
                    all_edges.append((sid, tid))

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

    # 1.2 Parse Author Data
    author_data = {}
    author_collab = {}

    for _, row in meta.iterrows():
        pid = str(row['paper_openalex_id'])
        if pid not in nodes_data.index:
            continue

        if pd.isna(row['author_id_list']):
            continue
        ids = str(row['author_id_list']).split(';')
        names = str(row['author_list']).split(';')

        p_x, p_y = nodes_data.at[pid, 'x'], nodes_data.at[pid, 'y']
        p_year, p_cite = row['publication_year'], row['cited_by_count']
        p_title = row['title']
        p_cluster = nodes_data.at[pid, 'cluster']  
        p_theta = np.arctan2(p_y, p_x) % (2 * np.pi)

        for i, aid in enumerate(ids):
            aid = aid.strip()
            if not aid:
                continue
            if aid not in author_data:
                author_data[aid] = {
                    'name': names[i].strip(), 
                    'years': [], 
                    'paper_thetas': [],
                    'cites': 0, 
                    'papers_built': [], 
                    'co_authors_set': set(),
                    'clusters': [],
                    'cluster_max_years': {}  
                }
            author_data[aid]['years'].append(p_year)
            author_data[aid]['paper_thetas'].append(p_theta)
            author_data[aid]['cites'] += p_cite
            author_data[aid]['clusters'].append(p_cluster)
            
            prev_max = author_data[aid]['cluster_max_years'].get(p_cluster, -1)
            author_data[aid]['cluster_max_years'][p_cluster] = max(prev_max, p_year)

            if p_title and p_title != 'Unknown':
                author_data[aid]['papers_built'].append({'title': p_title, 'cite': p_cite})

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                aid1, aid2 = ids[i].strip(), ids[j].strip()
                if aid1 and aid2:
                    pair = tuple(sorted([aid1, aid2]))
                    author_collab[pair] = author_collab.get(pair, 0) + 1

                    if aid1 in author_data:
                        author_data[aid1]['co_authors_set'].add(aid2)
                    if aid2 in author_data:
                        author_data[aid2]['co_authors_set'].add(aid1)

    # 1.3 Construct Author DataFrame & Author Edges
    R_MIN, R_MAX = 0.2, 1.0
    author_semantic_thetas = {}
    for aid, info in author_data.items():
        sin_sum = np.sum(np.sin(info['paper_thetas']))
        cos_sum = np.sum(np.cos(info['paper_thetas']))
        author_semantic_thetas[aid] = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)

    all_first_years = [np.min(info['years']) for info in author_data.values() if info['years']]
    global_min_first_yr = np.min(all_first_years) if all_first_years else min_yr
    global_max_first_yr = np.max(all_first_years) if all_first_years else max_yr

    author_rows = []
    for aid, info in author_data.items():
        first_year = np.min(info['years'])
        if global_max_first_yr == global_min_first_yr:
            r = (R_MIN + R_MAX) / 2
        else:
            norm_year = (first_year - global_min_first_yr) / (global_max_first_yr - global_min_first_yr)
            r = R_MIN + norm_year * (R_MAX - R_MIN)

        s_theta = author_semantic_thetas[aid]
        valid_co_thetas = [author_semantic_thetas[ca_id] for ca_id in info['co_authors_set'] if ca_id in author_semantic_thetas]
        if valid_co_thetas:
            sin_co = np.sum(np.sin(valid_co_thetas))
            cos_co = np.sum(np.cos(valid_co_thetas))
            c_theta = np.arctan2(sin_co, cos_co) % (2 * np.pi)
        else:
            c_theta = s_theta

        diff_cluster = np.arctan2(np.sin(c_theta - s_theta), np.cos(c_theta - s_theta))
        semantic_base = s_theta + WEIGHT_CLUSTER * diff_cluster
        curr_theta = semantic_base  
        diff_citation = np.arctan2(np.sin(curr_theta - semantic_base), np.cos(curr_theta - semantic_base))
        final_theta = (semantic_base + WEIGHT_CITATION * diff_citation) % (2 * np.pi)

        final_x = r * np.cos(final_theta)
        final_y = r * np.sin(final_theta)

        cluster_counts = Counter(info['clusters'])
        if cluster_counts:
            sorted_clusters = sorted(
                cluster_counts.keys(),
                key=lambda c: (-cluster_counts[c], -info['cluster_max_years'].get(c, 0))
            )
            main_cluster = sorted_clusters[0]
        else:
            main_cluster = 0    
            
        cluster_counts = dict(Counter(info['clusters']))
        cluster_counts = {int(k): int(v) for k, v in cluster_counts.items()}
        total_papers = len(info['years'])
        avg_cites = info['cites'] / total_papers if total_papers > 0 else 0

        top_papers = sorted(info['papers_built'], key=lambda x: x['cite'], reverse=True)[:3]
        papers_str = " ; ".join([f'"{p["title"]}" (Cites: {int(p["cite"])})' for p in top_papers]) if top_papers else "No records"
        co_names = [author_data[ca_id]['name'] for ca_id in info['co_authors_set'] if ca_id in author_data]
        co_authors_str = ", ".join(co_names[:5]) if co_names else "Mainly Independent Research"

        rich_note = f"📊 Total Citations: {int(info['cites'])}\n" \
                    f"👴 First Active Year (Entry): {int(first_year)}\n" \
                    f"🤝 Key Collaborators: {co_authors_str}\n" \
                    f"📄 Selected Publications: {papers_str}"
        
        formatted_counts = {str(k): v for k, v in cluster_counts.items()}
        author_rows.append({
            'author_id': aid, 'name': info['name'], 'note': rich_note,
            'x': final_x, 'y': final_y, 'publication_year': first_year,
            'cited_by_count': info['cites'], 
            'avg_cites': avg_cites, 
            'cluster_counts': json.dumps(formatted_counts),
            'cluster': main_cluster 
        })
        
    nodes_author = pd.DataFrame(author_rows)
    if not nodes_author.empty:
        nodes_author = nodes_author.set_index('author_id')

    edges_author = []
    for (aid1, aid2), weight in author_collab.items():
        if aid1 in nodes_author.index and aid2 in nodes_author.index:
            edges_author.append((aid1, aid2, weight))

    return nodes_data, all_edges, nodes_author, edges_author, meta, int(min_yr), int(max_yr)


# Initialize Data
nodes_df, edges_pool, nodes_author, edges_author, meta_df, MIN_Y, MAX_Y = load_and_layout()

# --- 2. Dash Layout ---
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

COLOR_PALETTE = ['#8B7E6F', '#B4C4D5', '#9E9E7E', '#A58B84', '#7E8B9E', '#D6DADB', '#4A453F', '#C2B49B']

# CSS Helper Styles
CARD_STYLE = {
    'backgroundColor': '#FFFFFF',
    'borderRadius': '8px',
    'padding': '12px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
    'border': '1px solid #E5E0D8',
    'marginBottom': '12px'
}

# App Layout Structure Refactored
app.layout = html.Div(style={
    'backgroundColor': '#F5F3EB',
    'minHeight': '100vh',
    'padding': '15px',
    'fontFamily': 'Inter, system-ui, sans-serif',
    'boxSizing': 'border-box'
}, children=[

    # Outer Grid Container
    html.Div(style={
        'display': 'grid',
        'gridTemplateColumns': '280px 1fr 340px',
        'gridTemplateRows': 'auto 1fr auto',
        'gap': '12px',
        'maxWidth': '1800px',
        'margin': '0 auto'
    }, children=[

        # ==========================================
        # TOP ROW: 2. Control Panel (跨越中间和右边/顶部)
        # ==========================================
        html.Div(style={
            'gridColumn': '1 / -1',
            'backgroundColor': '#FFFFFF',
            'borderRadius': '8px',
            'padding': '10px 18px',
            'border': '1px solid #E5E0D8',
            'boxShadow': '0 2px 6px rgba(0,0,0,0.04)',
            'display': 'flex',
            'alignItems': 'center',
            'justifySizing': 'space-between',
            'gap': '20px'
        }, children=[
            # 模式切换
            html.Div([
                html.Label("View Mode", style={'fontWeight': 'bold', 'color': '#4A453F', 'fontSize': '12px', 'display': 'block', 'marginBottom': '4px'}),
                dcc.RadioItems(
                    id='view-mode',
                    options=[
                        {'label': ' 引文网络 (Paper)', 'value': 'paper'},
                        {'label': ' 作者合作网络 (Author)', 'value': 'author'}
                    ],
                    value='paper',
                    labelStyle={'display': 'inline-block', 'marginRight': '12px', 'fontSize': '12px'}
                )
            ], style={'flex': '0 0 220px'}),

            # 年份范围
            html.Div([
                html.Label("Publication Year Range", style={'fontWeight': 'bold', 'color': '#4A453F', 'fontSize': '12px'}),
                dcc.RangeSlider(id='year-slider', min=MIN_Y, max=MAX_Y, step=1, value=[MIN_Y, MAX_Y],
                                marks={i: str(i) for i in range(MIN_Y, MAX_Y + 1, 3)})
            ], style={'flex': '1'}),

            # 节点大小与缩放
            html.Div([
                html.Label("Node Size / Citation Scaling", style={'fontWeight': 'bold', 'color': '#4A453F', 'fontSize': '12px'}),
                html.Div([
                    dcc.Slider(id='base-size-slider', min=1, max=20, step=0.5, value=5, tooltip={"placement": "bottom"}),
                    dcc.Slider(id='scale-factor-slider', min=0, max=100, step=5, value=35, tooltip={"placement": "bottom"})
                ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '10px'})
            ], style={'flex': '1'})
        ]),

        # ==========================================
        # LEFT COLUMN: 1. 模糊搜索功能 & 7. 两个悬浮/信息窗
        # ==========================================
        html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}, children=[
            
            # 1. 模糊搜索功能卡片
            html.Div(style=CARD_STYLE, children=[
                html.H4("🔍 Search & Filter", style={'margin': '0 0 8px 0', 'fontSize': '14px', 'color': '#4A453F'}),
                dcc.Input(id='search-box', type='text', placeholder='Search title, abstract keywords...',
                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #CCC', 'boxSizing': 'border-box'}),
                html.Div(id='search-results-list', style={'fontSize': '11px', 'color': '#666', 'marginTop': '6px'})
            ]),

            # 7. AI Search & Comparison 模块
            html.Div(id='ai-panel', style=CARD_STYLE, children=[
                html.Div(id='ai-panel-header', style={'marginBottom': '8px', 'paddingBottom': '4px', 'borderBottom': '1px solid #EEE'}, children=[
                    html.H4("🤖 AI Search & Comparison", style={'color': '#4A453F', 'margin': '0', 'fontSize': '13px', 'display': 'inline-block'}),
                ]),

                html.Div([
                    html.Span("Selected Objects (Max 2):", style={'fontSize': '11px', 'fontWeight': 'bold'}),
                    html.Div(id='selected-nodes-tags', style={'marginTop': '4px', 'marginBottom': '8px'})
                ]),

                html.Button("🧹 Clear Selection", id='clear-selection-btn', n_clicks=0,
                            style={'padding': '2px 6px', 'marginBottom': '8px', 'backgroundColor': '#E0DCD3', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '10px'}),

                dcc.Textarea(
                    id='ai-input',
                    placeholder='Ask a question...',
                    style={'width': '100%', 'height': '50px', 'borderRadius': '4px', 'borderColor': '#CCC', 'padding': '6px', 'fontSize': '11px', 'boxSizing': 'border-box'}
                ),

                html.Div([
                    html.Button("🚀 Ask AI", id='ask-ai-btn', n_clicks=0,
                                style={'padding': '4px 8px', 'backgroundColor': '#8B7E6F', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'marginRight': '6px', 'fontSize': '11px'}),
                    html.Button("⚖️ Smart Compare", id='compare-ai-btn', n_clicks=0,
                                style={'padding': '4px 8px', 'backgroundColor': '#7E8B9E', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '11px'}),
                ], style={'marginTop': '6px'}),

                dcc.Loading(
                    type="circle",
                    children=html.Div(id='ai-output', style={
                        'marginTop': '8px', 'padding': '8px', 'backgroundColor': '#F9F8F3',
                        'borderRadius': '4px', 'fontSize': '11px', 'lineHeight': '1.4', 'maxHeight': '150px', 'overflowY': 'auto'
                    }, children="Select nodes to compare or ask AI.")
                ),
                dcc.Store(id='selected-nodes-store', data=[])
            ]),

            # 7. 节点具体信息悬浮/固定窗
            html.Div(id='info-panel', style={**CARD_STYLE, 'flex': '1', 'overflowY': 'auto', 'minHeight': '200px'})
        ]),

        # ==========================================
        # CENTER COLUMN: 3. 主视图1 & 主视图2 (切换展示)
        # ==========================================
        html.Div(style={'position': 'relative', 'backgroundColor': '#FFFFFF', 'borderRadius': '8px', 'border': '1px solid #E5E0D8', 'padding': '10px'}, children=[
            dcc.Graph(id='main-plot', config={'displayModeBar': False}, style={'height': '68vh', 'width': '100%'}),

            # 浮动作者控制选项
            html.Div(
                id='author-size-container',
                children=[
                    html.Span("NODE SIZE METRIC", style={'fontSize': '9px', 'fontWeight': '700', 'color': '#8C8275', 'marginBottom': '4px', 'display': 'block'}),
                    dcc.RadioItems(
                        id='author-size-metric',
                        options=[{'label': ' Total Cites', 'value': 'total'}, {'label': ' Avg Cites', 'value': 'avg'}],
                        value='total',
                        labelStyle={'display': 'inline-block', 'marginRight': '8px', 'fontSize': '11px'}
                    )
                ],
                style={'position': 'absolute', 'bottom': '20px', 'left': '20px', 'zIndex': '10', 'backgroundColor': 'rgba(255,255,255,0.9)', 'padding': '6px 12px', 'borderRadius': '6px', 'border': '1px solid #CCC', 'display': 'none'}
            )
        ]),

        # ==========================================
        # RIGHT COLUMN: 4. 三类统计图表 (点击展开 / 三选二展示)
        # ==========================================
        html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}, children=[
            
            # Chart Group 1: Bibliometric Overview
            html.Details([
                html.Summary("📊 Bibliometric Overview", style={'fontWeight': 'bold', 'cursor': 'pointer', 'padding': '8px', 'backgroundColor': '#EFECE6', 'borderRadius': '4px', 'fontSize': '13px'}),
                html.Div([
                    html.Img(id='wordcloud-img', style={'width': '100%', 'height': 'auto', 'borderRadius': '4px'}),
                    dcc.Graph(id='concept-bar-plot', config={'displayModeBar': False}, style={'height': '180px'}),
                    dcc.Graph(id='evolution-river-graph', config={'displayModeBar': False}, style={'height': '200px'})
                ], style={'padding': '8px'})
            ], open=True, style=CARD_STYLE),

            # Chart Group 2: Research Scenario
            html.Details([
                html.Summary("📈 Research Scenario", style={'fontWeight': 'bold', 'cursor': 'pointer', 'padding': '8px', 'backgroundColor': '#EFECE6', 'borderRadius': '4px', 'fontSize': '13px'}),
                html.Div([
                    dcc.Graph(id='citation-histogram-graph', config={'displayModeBar': False}, style={'height': '220px'})
                ], style={'padding': '8px'})
            ], open=True, style=CARD_STYLE),

            # Chart Group 3: Modeling Strategy
            html.Details([
                html.Summary("⚙️ Modeling Strategy", style={'fontWeight': 'bold', 'cursor': 'pointer', 'padding': '8px', 'backgroundColor': '#EFECE6', 'borderRadius': '4px', 'fontSize': '13px'}),
                html.Div([
                    dcc.Graph(id='stacked-trend-graph', config={'displayModeBar': False}, style={'height': '220px'})
                ], style={'padding': '8px'})
            ], open=False, style=CARD_STYLE)
        ]),

        # ==========================================
        # BOTTOM ROW: 6. 引文主路径分析 & 5. 第一单位分布地图
        # ==========================================
        html.Div(style={'gridColumn': '1 / 3', **CARD_STYLE}, children=[
            html.H4("🛣️ 6. 引文主路径分析 (Main Path Analysis)", style={'margin': '0 0 8px 0', 'fontSize': '13px', 'color': '#4A453F'}),
            html.Div("Main Path Analysis Timeline Visualization", style={'height': '180px', 'backgroundColor': '#FAF8F5', 'border': '1px dashed #CCC', 'borderRadius': '4px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'color': '#888', 'fontSize': '12px'})
        ]),

        html.Div(style={'gridColumn': '3 / 4', **CARD_STYLE}, children=[
            html.H4("🌍 5. 第一单位分布地图 (Geospatial Distribution)", style={'margin': '0 0 8px 0', 'fontSize': '13px', 'color': '#4A453F'}),
            html.Iframe(id='geo-3d-map-iframe', style={'width': '100%', 'height': '180px', 'border': 'none', 'borderRadius': '4px'})
        ])

    ])
])

# --- 3. Interaction Callback Logic ---
@app.callback(
    Output('main-plot', 'figure'),
    [Input('view-mode', 'value'),
     Input('year-slider', 'value'),
     Input('search-box', 'value'),
     Input('base-size-slider', 'value'),
     Input('scale-factor-slider', 'value'),
     Input('author-size-metric', 'value')] 
)
def update_network(view_mode, years, search_txt, base_size, scale_factor, author_size_metric):
    if view_mode == 'paper':
        df = nodes_df
        edges_pool_to_use = edges_pool
        label_col = 'title'
    else:
        df = nodes_author
        edges_pool_to_use = edges_author
        label_col = 'name'

    filtered_nodes = df[(df['publication_year'] >= years[0]) & (df['publication_year'] <= years[1])].copy()
    node_ids = set(filtered_nodes.index)

    if filtered_nodes.empty:
        return go.Figure()

    cites_col = 'cited_by_count'
    if view_mode == 'author' and author_size_metric == 'avg':
        cites_col = 'avg_cites'

    sqrt_cites = np.sqrt(filtered_nodes[cites_col].astype(float))
    max_sqrt = sqrt_cites.max() if sqrt_cites.max() > 0 else 1.0
    filtered_nodes['node_s'] = base_size + (sqrt_cites / max_sqrt) * scale_factor

    if view_mode == 'paper':
        node_colors = [COLOR_PALETTE[int(c) % 8] for c in filtered_nodes['cluster']]
        custom_data_arr = np.stack((filtered_nodes['abstract'], filtered_nodes.index, [''] * len(filtered_nodes)), axis=-1)
    else:
        node_colors = [COLOR_PALETTE[int(c) % 8] for c in filtered_nodes['cluster']]
        custom_data_arr = np.stack((filtered_nodes['note'], filtered_nodes.index, filtered_nodes['cluster_counts']), axis=-1)

    current_edges = []
    for edge in edges_pool_to_use:
        u, v = edge[0], edge[1]
        if u in node_ids and v in node_ids:
            current_edges.append((u, v))

    edge_x, edge_y = [], []
    if current_edges:
        nodes_for_hb = filtered_nodes[['x', 'y']]
        edges_for_hb = pd.DataFrame(current_edges, columns=['source', 'target'])
        hb_paths = hammer_bundle(nodes_for_hb, edges_for_hb, initial_bandwidth=0.1, decay=0.7)
        edge_x = hb_paths['x'].tolist()
        edge_y = hb_paths['y'].tolist()

    marker_line_widths = [0] * len(filtered_nodes)
    if search_txt and len(search_txt) > 1:
        highlight_mask = filtered_nodes[label_col].str.contains(search_txt, case=False, na=False)
        marker_line_widths = [2.5 if is_match else 0 for is_match in highlight_mask]
        
    fig = go.Figure()

    for y_val in np.linspace(MIN_Y, MAX_Y, 6):
        r_val = 0.2 + 0.8 * (y_val - MIN_Y) / (MAX_Y - MIN_Y + 1e-5)
        fig.add_shape(type="circle", xref="x", yref="y", x0=-r_val, y0=-r_val, x1=r_val, y1=r_val,
                      line=dict(color="#D6DADB", width=1, dash="dot"))

    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.6, color='#4A453F'), hoverinfo='none', mode='lines', opacity=0.2))

    fig.add_trace(go.Scatter(
        x=filtered_nodes['x'], y=filtered_nodes['y'],
        mode='markers',
        text=filtered_nodes[label_col],
        customdata=custom_data_arr,
        marker=dict(
            size=filtered_nodes['node_s'],
            color=node_colors,
            line=dict(width=marker_line_widths, color='red'),
            opacity=0.8
        ),
        hoverinfo='text'
    ))

    fig.update_layout(
        showlegend=False, clickmode='event',
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2], fixedrange=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2], scaleanchor="x", scaleratio=1, fixedrange=False)
    )
    return fig

@app.callback(
    Output('search-box', 'placeholder'),
    [Input('view-mode', 'value')]
)
def update_search_placeholder(view_mode):
    if view_mode == 'author':
        return 'Filter by author name...'
    return 'Search title, abstract keywords...'

@app.callback(
    Output('info-panel', 'children'),
    [Input('main-plot', 'clickData')],
    [State('view-mode', 'value')],
    prevent_initial_call=False
)
def handle_click(clickData, view_mode):
    if not clickData or 'points' not in clickData or 'customdata' not in clickData['points'][0]:
        return html.Div("💡 Click on any node to view detailed metrics and information.", style={'color': '#888', 'fontSize': '11px', 'fontStyle': 'italic'})

    point = clickData['points'][0]
    title = point.get('text', 'Unknown')
    custom_data = point.get('customdata', [])
    
    info = custom_data[0] if len(custom_data) > 0 else ""
    cluster_counts_str = custom_data[2] if len(custom_data) > 2 else ""

    if view_mode == 'author':
        lines = info.split('\n') if info else []
        line_0 = lines[0] if len(lines) > 0 else ""
        line_1 = lines[1] if len(lines) > 1 else ""
        line_2 = lines[2] if len(lines) > 2 else ""
        line_3 = lines[3] if len(lines) > 3 else ""
        
        pie_graph = html.Div()
        if cluster_counts_str:
            try:
                counts = json.loads(cluster_counts_str)
                labels = [f"Topic {k}" for k in counts.keys()]
                values = list(counts.values())
                pie_colors = [COLOR_PALETTE[int(k) % 8] for k in counts.keys()]
                
                pie_fig = go.Figure(data=[go.Pie(
                    labels=labels, values=values, hole=0.4,
                    marker=dict(colors=pie_colors),
                    textinfo='percent', hoverinfo='label+value'
                )])
                pie_fig.update_layout(
                    margin=dict(t=10, b=0, l=0, r=0), height=140,
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                )
                pie_graph = dcc.Graph(figure=pie_fig, config={'displayModeBar': False})
            except Exception:
                pie_graph = html.Div()

        panel_content = html.Div([
            html.H4(f"👤 {title}", style={'color': '#4A453F', 'fontSize': '14px', 'margin': '0 0 6px 0'}),
            pie_graph,
            html.Div([
                html.P(line_0, style={'margin': '3px 0', 'fontSize': '11px'}),
                html.P(line_1, style={'margin': '3px 0', 'fontSize': '11px'}),
                html.P(line_2, style={'margin': '3px 0', 'color': '#666', 'fontSize': '11px'}),
            ], style={'backgroundColor': '#F9F8F3', 'padding': '6px', 'borderRadius': '4px'}),
            html.Strong("🎓 Selected Works:", style={'fontSize': '11px', 'color': '#4A453F', 'marginTop': '6px', 'display': 'block'}),
            html.P(line_3.replace("Selected Publications: ", ""), style={'fontSize': '10px', 'lineHeight': '1.4', 'color': '#555', 'marginTop': '4px'})
        ])

    else:
        paper_id = custom_data[1] if len(custom_data) > 1 else 'N/A'
        abstract = info if info else "No abstract available."

        panel_content = html.Div([
            html.H4(f"📄 {title}", style={'color': '#4A453F', 'fontSize': '13px', 'margin': '0 0 6px 0'}),
            html.P(f"ID: {paper_id}", style={'margin': '2px 0', 'fontSize': '10px', 'color': '#7F8C8D'}),
            html.Strong("Abstract:", style={'fontSize': '11px', 'color': '#4A453F', 'marginTop': '6px', 'display': 'block'}),
            html.P(abstract, style={'fontSize': '10px', 'lineHeight': '1.4', 'color': '#555', 'marginTop': '4px'})
        ])

    return panel_content

@app.callback(
    [Output('selected-nodes-store', 'data'),
     Output('selected-nodes-tags', 'children')],
    [Input('main-plot', 'clickData'),
     Input('clear-selection-btn', 'n_clicks')],
    [State('selected-nodes-store', 'data'),
     State('view-mode', 'value')]
)
def manage_selected_nodes(clickData, clear_clicks, current_selected, view_mode):
    if current_selected is None:
        current_selected = []

    if ctx.triggered_id == 'clear-selection-btn':
        return [], html.Span("💡 Click graph nodes to compare", style={'color': '#999', 'fontSize': '11px'})

    if ctx.triggered_id != 'main-plot' or not clickData:
        if not current_selected:
            return [], html.Span("💡 Click graph nodes to compare", style={'color': '#999', 'fontSize': '11px'})

        tags = [html.Span(f"{'📄' if n.get('type')=='paper' else '👤'} {n['name'][:8]}...", title=n['name'],
                          style={'display': 'inline-block', 'margin': '2px', 'padding': '2px 6px', 'backgroundColor': '#E0DCD3', 'borderRadius': '3px', 'fontSize': '10px'}) for n in current_selected]
        return current_selected, tags

    try:
        point = clickData['points'][0]
        node_name = point.get('text', 'Unknown')
        custom_data_list = point.get('customdata', [None, None])
        node_info = custom_data_list[0]
        node_id = str(custom_data_list[1])

        node_meta = {
            'id': node_id,
            'name': node_name,
            'type': view_mode,
            'info': node_info if node_info else "No abstract information available"
        }

        if not any(str(n['id']) == node_id for n in current_selected):
            current_selected = (current_selected + [node_meta])[-2:]

        tags = [html.Span(f"{'📄' if n['type']=='paper' else '👤'} {n['name'][:8]}...", title=n['name'],
                          style={'display': 'inline-block', 'margin': '2px', 'padding': '2px 6px', 'backgroundColor': '#C2B49B', 'borderRadius': '3px', 'fontSize': '10px', 'fontWeight': 'bold'}) for n in current_selected]

        return current_selected, tags
    except Exception as e:
        return current_selected, []

@app.callback(
    Output('ai-output', 'children'),
    [Input('ask-ai-btn', 'n_clicks'),
     Input('compare-ai-btn', 'n_clicks')],
    [State('selected-nodes-store', 'data'),
     State('ai-input', 'value')]
)
def handle_ai_query(ask_clicks, compare_clicks, selected_nodes, user_question):
    if not ctx.triggered:
        return "Select papers/authors to interact."

    trigger_id = ctx.triggered_id

    if not selected_nodes:
        return "❌ Select at least one object."

    if trigger_id == 'compare-ai-btn':
        if len(selected_nodes) < 2:
            return "❌ Requires 2 objects."
        return llm_helper.handle_ai_compare(selected_nodes[0], selected_nodes[1])

    elif trigger_id == 'ask-ai-btn':
        return llm_helper.handle_ai_question(selected_nodes, user_question)

    return "Awaiting command..."

@app.callback(
    [Output('wordcloud-img', 'src'),
     Output('concept-bar-plot', 'figure')],
    [Input('year-slider', 'value'),
     Input('view-mode', 'value')]
)
def update_wordcloud(years, view_mode):
    df_filtered = nodes_df[(nodes_df['publication_year'] >= years[0]) & (nodes_df['publication_year'] <= years[1])]
    img_src, bar_fig = get_wordcloud_and_bar_assets(df_filtered, target_col='concepts')
    return img_src, bar_fig

@app.callback(
    Output('evolution-river-graph', 'figure'),
    [Input('year-slider', 'value')]
)
def update_evolution_river(years):
    return evolution_river.generate_evolution_river(meta_df, nodes_df, year_range=years)

@app.callback(
    Output('citation-histogram-graph', 'figure'),
    [Input('year-slider', 'value'),
     Input('view-mode', 'value')]
)
def update_citation_histogram(years, view_mode):
    return citation_histogram.generate_citation_histogram(nodes_df, year_range=years)

@app.callback(
    Output('stacked-trend-graph', 'figure'),
    [Input('year-slider', 'value')]
)
def update_stacked_trend(years):
    return stacked_trend.generate_stacked_trend_figure(stacked_data_df, year_range=years)

@app.callback(
    Output('geo-3d-map-iframe', 'srcDoc'), 
    [Input('year-slider', 'value')]
)
def update_geo_3d_map(years):
    return geo_map_3d.generate_3d_column_map_html(stacked_data_df, year_range=years, gps_col='first_affil_city_gps')

@app.callback(
    Output('author-size-container', 'style'),
    Input('view-mode', 'value')
)
def toggle_author_size_control(view_mode):
    style = {'position': 'absolute', 'bottom': '20px', 'left': '20px', 'zIndex': '10', 'backgroundColor': 'rgba(255,255,255,0.9)', 'padding': '6px 12px', 'borderRadius': '6px', 'border': '1px solid #CCC'}
    style['display'] = 'block' if view_mode == 'author' else 'none'
    return style

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8051, debug=False)