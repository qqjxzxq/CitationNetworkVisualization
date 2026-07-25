from collections import Counter
import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, clientside_callback  # 👈 确保导入 clientside_callback
from datashader.bundling import hammer_bundle
import llm_helper
from wordcloud_module import get_wordcloud_and_bar_assets


# --- 1. Data Preprocessing (Fixed Coordinates Calculation) ---
def load_and_layout():
    # Load Data using absolute paths for deployment safety
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(current_dir, 'sample.csv')
    umap_path = os.path.join(current_dir, 'abstract_umap.csv')

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

    # Extract all edges
    all_edges = []
    for _, row in meta.iterrows():
        sid = str(row['paper_openalex_id'])
        if pd.notna(row['referenced_ids_openalex']):
            targets = str(row['referenced_ids_openalex']).replace(';', '@').split('@')
            for tid in [t.strip() for t in targets]:
                if sid in nodes_data.index and tid in nodes_data.index:
                    all_edges.append((sid, tid))

    # Iterative coordinate adjustment
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

        for i, aid in enumerate(ids):
            aid = aid.strip()
            if not aid:
                continue
            if aid not in author_data:
                author_data[aid] = {
                    'name': names[i].strip(), 'years': [], 'xs': [], 'ys': [],
                    'cites': 0, 'papers_built': [], 'co_authors_set': set(),
                    'clusters': []  
                }
            author_data[aid]['xs'].append(p_x)
            author_data[aid]['ys'].append(p_y)
            author_data[aid]['years'].append(p_year)
            author_data[aid]['cites'] += p_cite
            author_data[aid]['clusters'].append(p_cluster)

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

    # 1.3 Construct Author DataFrame
    author_rows = []
    for aid, info in author_data.items():
        main_cluster = Counter(info['clusters']).most_common(1)[0][0] if info['clusters'] else 0

        top_papers = sorted(info['papers_built'], key=lambda x: x['cite'], reverse=True)[:3]
        papers_str = " ; ".join([f'"{p["title"]}" (Cites: {int(p["cite"])})' for p in top_papers]) if top_papers else "No records"

        co_names = [author_data[ca_id]['name'] for ca_id in info['co_authors_set'] if ca_id in author_data]
        co_authors_str = ", ".join(co_names[:5]) if co_names else "Mainly Independent Research"

        rich_note = f"📊 Total Citations: {int(info['cites'])}\n" \
                    f"📅 First Active Year: {int(np.min(info['years']))}\n" \
                    f"🤝 Key Collaborators: {co_authors_str}\n" \
                    f"📄 Selected Publications: {papers_str}"

        author_rows.append({
            'author_id': aid,
            'name': info['name'],
            'note': rich_note,
            'x': np.mean(info['xs']),
            'y': np.mean(info['ys']),
            'publication_year': np.min(info['years']),
            'cited_by_count': info['cites'],
            'cluster': main_cluster 
        })
    nodes_author = pd.DataFrame(author_rows).set_index('author_id')

    # 1.4 Extract Author Edges
    edges_author = [list(p) for p in author_collab.keys() if p[0] in nodes_author.index and p[1] in nodes_author.index]

    return nodes_data, all_edges, nodes_author, edges_author, int(min_yr), int(max_yr)


# Initialize Data
nodes_df, edges_pool, nodes_author, edges_author, MIN_Y, MAX_Y = load_and_layout()

# --- 2. Dash Layout ---
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

COLOR_PALETTE = ['#8B7E6F', '#B4C4D5', '#9E9E7E', '#A58B84', '#7E8B9E', '#D6DADB', '#4A453F', '#C2B49B']

app.layout = html.Div(style={'backgroundColor': '#F2F0E4', 'minHeight': '100vh', 'padding': '20px'}, children=[
    html.H2("Citation Network - Interactive Visualization", style={'textAlign': 'center', 'color': '#4A453F', 'marginBottom': '20px'}),

    # Control Panel
    html.Div([
        html.Label("🌐 View Mode:", style={'fontWeight': 'bold', 'color': '#4A453F'}),
        dcc.RadioItems(
            id='view-mode',
            options=[
                {'label': ' Paper Citation Network', 'value': 'paper'},
                {'label': ' Author Collaboration Network', 'value': 'author'}
            ],
            value='paper',
            labelStyle={'display': 'inline-block', 'marginRight': '20px', 'marginTop': '5px'}
        )
    ], style={'marginBottom': '20px', 'paddingBottom': '15px', 'borderBottom': '1px solid #eee'}),

    html.Div([
        # Row 1: Search & Year Range
        html.Div([
            html.Div([
                html.Label("🔍 Search:", style={'fontWeight': 'bold'}),
                dcc.Input(id='search-box', type='text', placeholder='Title keywords...',
                          style={'width': '90%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
            ], style={'width': '33%', 'display': 'inline-block'}),
            html.Div([
                html.Label("📅 Publication Year Range:", style={'fontWeight': 'bold'}),
                dcc.RangeSlider(id='year-slider', min=MIN_Y, max=MAX_Y, step=1, value=[MIN_Y, MAX_Y],
                                marks={i: str(i) for i in range(MIN_Y, MAX_Y + 1, 5)})
            ], style={'width': '66%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'marginBottom': '20px'}),

        # Row 2: Node Size Controls
        html.Div([
            html.Div([
                html.Label("🔘 Base Size:", style={'fontWeight': 'bold'}),
                dcc.Slider(id='base-size-slider', min=1, max=20, step=0.5, value=5,
                           tooltip={"placement": "bottom", "always_visible": True})
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                html.Label("🚀 Citation Scaling Factor:", style={'fontWeight': 'bold'}),
                dcc.Slider(id='scale-factor-slider', min=0, max=100, step=5, value=35,
                           tooltip={"placement": "bottom", "always_visible": True})
            ], style={'width': '48%', 'display': 'inline-block'})
        ])
    ], style={'background': 'white', 'padding': '20px', 'borderRadius': '10px',
              'boxShadow': '0 2px 10px rgba(0,0,0,0.05)', 'marginBottom': '20px'}),

    # Main Plotting Area
    html.Div([
        # 📌 1. AI Research Assistant Panel (保持原悬浮窗样式 + 增加可拖拽 Header)
        html.Div(id='ai-panel', style={
            'position': 'absolute', 'top': '20px', 'left': '20px', 'width': '260px',
            'backgroundColor': 'rgba(255, 255, 255, 0.95)', 'padding': '20px',
            'borderRadius': '12px', 'boxShadow': '0 8px 30px rgba(0,0,0,0.15)',
            'display': 'block', 'maxHeight': '70vh', 'overflowY': 'auto',
            'border': '1px solid #8B7E6F', 'zIndex': '1000', 'textAlign': 'left'
        }, children=[
            # 💡 给标题区域加上 id='ai-panel-header'，并指定 'cursor': 'move' 手势，作为按住拖拽的手柄
            html.Div(id='ai-panel-header', style={
                'cursor': 'move', 'userSelect': 'none', 'marginBottom': '15px', 'borderBottom': '1px solid #eee', 'paddingBottom': '5px'
            }, children=[
                html.H3("🤖 AI Search & Comparison", style={'color': '#4A453F', 'margin': '0', 'fontSize': '16px', 'display': 'inline-block'}),
                html.Span(" ⠿", style={'color': '#999', 'fontSize': '14px', 'marginLeft': '5px'})
            ]),

            # Selected Items Container
            html.Div([
                html.Strong("Selected Objects (Max 2):", style={'fontSize': '12px'}),
                html.Div(id='selected-nodes-tags', style={'marginTop': '5px', 'marginBottom': '10px'})
            ]),

            # Clear Selection Button
            html.Button("🧹 Clear Selection", id='clear-selection-btn', n_clicks=0,
                        style={'padding': '3px 8px', 'marginBottom': '12px', 'backgroundColor': '#D6DADB', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '11px'}),

            # Dialogue Input Area
            dcc.Textarea(
                id='ai-input',
                placeholder='Ask a question... (If 2 objects selected, click Smart Compare directly)',
                style={'width': '93%', 'height': '60px', 'borderRadius': '4px', 'borderColor': '#ccc', 'padding': '6px', 'fontFamily': 'inherit', 'fontSize': '12px'}
            ),

            html.Div([
                html.Button("🚀 Ask AI", id='ask-ai-btn', n_clicks=0,
                            style={'padding': '6px 12px', 'backgroundColor': '#8B7E6F', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'marginRight': '10px', 'fontSize': '12px'}),
                html.Button("⚖️ Smart Compare", id='compare-ai-btn', n_clicks=0,
                            style={'padding': '6px 12px', 'backgroundColor': '#7E8B9E', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '12px'}),
            ], style={'marginTop': '8px'}),

            # AI Output Display
            dcc.Loading(
                type="circle",
                children=html.Div(id='ai-output', style={
                    'marginTop': '15px', 'padding': '10px', 'backgroundColor': '#F9F8F3',
                    'borderRadius': '6px', 'borderLeft': '4px solid #8B7E6F', 'fontSize': '12px',
                    'lineHeight': '1.6', 'whiteSpace': 'pre-line', 'textAlign': 'justify'
                }, children="Click nodes in the graph to add them to the AI comparison queue.")
            ),

            # Hidden Store
            dcc.Store(id='selected-nodes-store', data=[])
        ]),

        # 📌 2. Main Graph
        dcc.Graph(id='main-plot', config={'displayModeBar': False},
                  style={'height': '80vh', 'width': '80vh', 'margin': '0 auto'}),

        # 📌 3. Dynamic Concepts Wordcloud Section (已修改为左右双栏布局)
        html.Div([
            html.H3("🔤 Academic Concept Evolution",
                    style={'color': '#4A453F', 'fontSize': '16px', 'marginBottom': '15px', 'textAlign': 'center'}),
            
            # 左右分栏容器
            html.Div([
                # 左侧：词云图片展示区
                html.Div([
                    html.Img(id='wordcloud-img', style={'width': '100%', 'height': 'auto', 'borderRadius': '6px'})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '2%'}),

                # 右侧：Top 10 频次柱状图
                html.Div([
                    dcc.Graph(id='concept-bar-plot', config={'displayModeBar': False}, style={'height': '350px'})
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ])
        ], style={
            'background': '#F2F0E4', 
            'padding': '20px', 'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.05)', 'marginTop': '20px'
        }),

        # 📌 4. Right Information Detail Panel
        html.Div(id='info-panel', style={
            'position': 'absolute', 'top': '20px', 'right': '20px', 'width': '340px',
            'backgroundColor': 'rgba(255, 255, 255, 0.95)', 'padding': '20px',
            'borderRadius': '8px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.15)',
            'display': 'none', 'maxHeight': '80vh', 'overflowY': 'auto', 'border': '1px solid #8B7E6F', 'zIndex': '1000',
            'textAlign': 'left'
        })
    ], style={'position': 'relative', 'textAlign': 'center'})
])


# --- 3. Interaction Callback Logic ---
@app.callback(
    Output('main-plot', 'figure'),
    [Input('view-mode', 'value'),
     Input('year-slider', 'value'),
     Input('search-box', 'value'),
     Input('base-size-slider', 'value'),
     Input('scale-factor-slider', 'value')]
)
def update_network(view_mode, years, search_txt, base_size, scale_factor):
    if view_mode == 'paper':
        df = nodes_df
        edges_pool_to_use = edges_pool
        label_col = 'title'
    else:
        df = nodes_author
        edges_pool_to_use = edges_author
        label_col = 'name'

    # 1. Filter Nodes
    filtered_nodes = df[(df['publication_year'] >= years[0]) & (df['publication_year'] <= years[1])].copy()
    node_ids = set(filtered_nodes.index)

    # 2. Dynamically Calculate Node Size
    sqrt_cites = np.sqrt(filtered_nodes['cited_by_count'])
    filtered_nodes['node_s'] = base_size + (sqrt_cites / (sqrt_cites.max() + 1)) * scale_factor

    # 3. Edge Bundling Calculation
    current_edges = [(u, v) for u, v in edges_pool_to_use if u in node_ids and v in node_ids]
    edge_x, edge_y = [], []
    if current_edges:
        nodes_for_hb = filtered_nodes[['x', 'y']]
        edges_for_hb = pd.DataFrame(current_edges, columns=['source', 'target'])
        hb_paths = hammer_bundle(nodes_for_hb, edges_for_hb, initial_bandwidth=0.1, decay=0.7)
        edge_x = hb_paths['x'].tolist()
        edge_y = hb_paths['y'].tolist()

    # 4. Search Highlight
    marker_line_widths = [0] * len(filtered_nodes)
    if search_txt and len(search_txt) > 1:
        highlight_idx = filtered_nodes[label_col].str.contains(search_txt, case=False, na=False)
        marker_line_widths = [2.5 if val else 0 for val in highlight_idx]

    # 5. Build Figure
    fig = go.Figure()

    # Background Concentric Circles
    for y_val in np.linspace(MIN_Y, MAX_Y, 6):
        r_val = 0.2 + 0.8 * (y_val - MIN_Y) / (MAX_Y - MIN_Y + 1e-5)
        fig.add_shape(type="circle", xref="x", yref="y", x0=-r_val, y0=-r_val, x1=r_val, y1=r_val,
                      line=dict(color="#D6DADB", width=1, dash="dot"))

    # Edges
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.6, color='#4A453F'),
                             hoverinfo='none', mode='lines', opacity=0.2))

    # Nodes
    fig.add_trace(go.Scatter(
        x=filtered_nodes['x'], y=filtered_nodes['y'],
        mode='markers',
        text=filtered_nodes[label_col],
        customdata=np.stack((
            filtered_nodes['abstract'] if view_mode == 'paper' else filtered_nodes['note'],
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


# Dynamically update search input placeholder based on mode
@app.callback(
    Output('search-box', 'placeholder'),
    [Input('view-mode', 'value')]
)
def update_search_placeholder(view_mode):
    if view_mode == 'author':
        return 'Filter by author name...'
    return 'Search title, abstract keywords...'


# Handle Click Events & Information Panel Rendering
@app.callback(
    [Output('info-panel', 'children'), Output('info-panel', 'style')],
    [Input('main-plot', 'clickData')],
    [State('view-mode', 'value')],
    prevent_initial_call=False
)
def handle_click(clickData, view_mode):
    if not clickData or 'points' not in clickData or 'customdata' not in clickData['points'][0]:
        return "", {'display': 'none'}

    point = clickData['points'][0]
    title = point['text']
    info = point['customdata'][0]

    if view_mode == 'author':
        lines = info.split('\n')
        panel_content = html.Div([
            html.H3(f"👤 {title}", style={'color': '#4A453F', 'fontSize': '18px', 'borderBottom': '2px solid #C2B49B', 'paddingBottom': '10px', 'marginTop': '5px'}),
            html.Div([
                html.P(lines[0], style={'margin': '6px 0', 'fontSize': '13px'}),
                html.P(lines[1], style={'margin': '6px 0', 'fontSize': '13px'}),
                html.P(lines[2], style={'margin': '6px 0', 'color': '#666', 'fontSize': '13px'}),
            ], style={'backgroundColor': '#F9F8F3', 'padding': '10px', 'borderRadius': '6px', 'marginBottom': '12px'}),

            html.Strong("🎓 Selected Works (Ranked by Citations):", style={'fontSize': '13px', 'color': '#4A453F'}),
            html.P(lines[3].replace("Selected Publications: ", ""), style={'fontSize': '12px', 'lineHeight': '1.6', 'color': '#555', 'marginTop': '6px', 'textAlign': 'justify'}),
            html.Hr(style={'borderColor': '#eee', 'margin': '15px 0'}),
            html.Em("Tip: Click blank area in graph to close this panel.", style={'fontSize': '11px', 'color': '#999'})
        ])
    else:
        panel_content = html.Div([
            html.H3(f"📄 {title}", style={'color': '#4A453F', 'fontSize': '15px', 'borderBottom': '2px solid #B4C4D5', 'paddingBottom': '10px', 'marginTop': '5px'}),
            html.P([
                html.Strong("🔍 Abstract: "),
                html.Span(info)
            ], style={'fontSize': '13px', 'lineHeight': '1.6', 'textAlign': 'justify', 'color': '#4A453F'}),
            html.Hr(style={'borderColor': '#eee', 'margin': '15px 0'}),
            html.Em("Tip: Click blank area in graph to close this panel.", style={'fontSize': '11px', 'color': '#999'})
        ])

    panel_style = {
        'position': 'absolute', 'top': '20px', 'right': '20px', 'width': '340px',
        'backgroundColor': 'rgba(255, 255, 255, 0.98)', 'padding': '20px',
        'borderRadius': '12px', 'boxShadow': '0 8px 30px rgba(0,0,0,0.18)',
        'display': 'block', 'maxHeight': '80vh', 'overflowY': 'auto',
        'border': '1px solid #8B7E6F', 'zIndex': '1000'
    }
    return panel_content, panel_style


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
        return [], html.Span("💡 Click graph nodes to add to comparison", style={'color': '#999', 'fontSize': '12px'})

    if ctx.triggered_id != 'main-plot' or not clickData:
        if not current_selected:
            return [], html.Span("💡 Click graph nodes to add to comparison", style={'color': '#999', 'fontSize': '12px'})

        tags = []
        for n in current_selected:
            icon = "📄" if n.get('type') == 'paper' else "👤"
            color = '#B4C4D5' if n.get('type') == 'paper' else '#C2B49B'
            tags.append(html.Span(f"{icon} {n['name'][:10]}...", title=n['name'],
                                  style={'display': 'inline-block', 'margin': '2px', 'padding': '4px 8px', 'backgroundColor': color, 'borderRadius': '4px', 'fontSize': '12px', 'color': '#4A453F'}))
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

        if any(str(n['id']) == node_id for n in current_selected):
            new_selected = current_selected
        else:
            new_selected = current_selected + [node_meta]
            if len(new_selected) > 2:
                new_selected = new_selected[1:]

        tags = []
        for n in new_selected:
            icon = "📄" if n['type'] == 'paper' else "👤"
            color = '#B4C4D5' if n['type'] == 'paper' else '#C2B49B'
            tags.append(html.Span(f"{icon} {n['name'][:10]}...", title=n['name'],
                                  style={'display': 'inline-block', 'margin': '2px', 'padding': '4px 8px', 'backgroundColor': color, 'borderRadius': '4px', 'fontSize': '12px', 'color': '#4A453F', 'fontWeight': 'bold'}))

        return new_selected, tags

    except Exception as e:
        print(f"Error parsing node: {str(e)}")
        return current_selected, []


# LLM Query Callback
@app.callback(
    Output('ai-output', 'children'),
    [Input('ask-ai-btn', 'n_clicks'),
     Input('compare-ai-btn', 'n_clicks')],
    [State('selected-nodes-store', 'data'),
     State('ai-input', 'value')]
)
def handle_ai_query(ask_clicks, compare_clicks, selected_nodes, user_question):
    if not ctx.triggered:
        return "No interaction data. Please click to select papers or authors in the graph."

    trigger_id = ctx.triggered_id

    if not selected_nodes:
        return "❌ Please click to select at least one paper or author in the graph!"

    if trigger_id == 'compare-ai-btn':
        if len(selected_nodes) < 2:
            return "❌ Comparison mode requires selecting 2 objects (two papers or two authors). Please click to select a second object and try again."

        return llm_helper.handle_ai_compare(selected_nodes[0], selected_nodes[1])

    elif trigger_id == 'ask-ai-btn':
        return llm_helper.handle_ai_question(selected_nodes, user_question)

    return "Awaiting command..."


# --- Dynamic Wordcloud Callback ---

@app.callback(
    [Output('wordcloud-img', 'src'),
     Output('concept-bar-plot', 'figure')],
    [Input('year-slider', 'value'),
     Input('view-mode', 'value')]
)
def update_wordcloud(years, view_mode):
    # 按年份过滤数据
    df_filtered = nodes_df[(nodes_df['publication_year'] >= years[0]) & (nodes_df['publication_year'] <= years[1])]
    
    # 直接调用模块生成 Base64 图片与 Plotly Bar Figure
    img_src, bar_fig = get_wordcloud_and_bar_assets(df_filtered, target_col='concepts')
    
    return img_src, bar_fig


# -------------------------------------------------------------
# 📌 4. Client-side Drag Callback (实现鼠标移动拖拽的核心)
# -------------------------------------------------------------
clientside_callback(
    """
    function(id) {
        setTimeout(function() {
            var panel = document.getElementById('ai-panel');
            var header = document.getElementById('ai-panel-header');
            if (panel && header && !panel.dataset.dragInited) {
                panel.dataset.dragInited = "true";
                var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
                header.onmousedown = function(e) {
                    e = e || window.event;
                    e.preventDefault();
                    pos3 = e.clientX;
                    pos4 = e.clientY;
                    document.onmouseup = function() {
                        document.onmouseup = null;
                        document.onmousemove = null;
                    };
                    document.onmousemove = function(e) {
                        e = e || window.event;
                        e.preventDefault();
                        pos1 = pos3 - e.clientX;
                        pos2 = pos4 - e.clientY;
                        pos3 = e.clientX;
                        pos4 = e.clientY;
                        panel.style.top = (panel.offsetTop - pos2) + "px";
                        panel.style.left = (panel.offsetLeft - pos1) + "px";
                    };
                };
            }
        }, 300);
        return "";
    }
    """,
    Output('ai-panel-header', 'title'),
    Input('ai-panel', 'id')
)


if __name__ == '__main__':
    # Local Debug
    # app.run(debug=True)

    # Web Deployment
    app.run(host='0.0.0.0', port=8051, debug=False)