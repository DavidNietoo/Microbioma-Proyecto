import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback_context, dash_table, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import io
import dash_uploader as du
from comandos_wsl import run_metaphlan_analysis
import plotly.colors as pc
import json
import plotly.graph_objs as go
import logging

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Configuracion del Dash Uploader
du.configure_upload(app, "uploads", use_upload_id=True)

app.uploaded_files = []

# Creacion del layout de la pagina de subida de archivos
upload_layout = dbc.Container([
   dbc.Row([
       dbc.Col([
        html.H2("Microbiome analysis using MetaPhlAn4"),
        html.H5("David Nieto Garza & Dr. Patricio Adrián Zapata Morín"),
        html.H4("Microbiome Analysis Upload"),
        html.P("Please upload your input files to start the analysis."),
       ], width=True),
   ], align="end"),
    du.Upload(
        id='dash-uploader',
        max_file_size=1800,  # 1800 MB
        filetypes=['fasta', 'fastq', 'bz2'],
        upload_id='upload',
        default_style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        text="Drag and Drop or Click to Upload Files",
        max_files=5  # maximo de archivos que se pueden subir
    ),
    html.Div(id='upload-status'),
    html.Div(id='uploaded-files-list'),  # Componente para mostrar los archivos subidos
    html.Button('Start Analysis', id='start-analysis-button', n_clicks=0, style={'margin': '10px'}),
    html.Div(id='analysis-status'),
    dcc.Loading(
        id="loading",
        type="default",
        children=html.Div(id="loading-output")
    ),
    html.Div(id='analysis-progress'),
])

# Creacion del layout de la pagina de analisis
analysis_layout = dbc.Container([
   dbc.Row([
       dbc.Col([
           html.H2("Microbiome analysis using MetaPhlAn4"),
           html.H5("David Nieto Garza & Dr. Patricio Adrián Zapata Morín"),
       ], width=True),
   ], align="end"),
   html.Hr(),
   dcc.Download(id="download"),
   dcc.Tabs([
       dcc.Tab(label='Graphs', children=[
           dbc.Row([
               dbc.Col([
                   html.Div([
                       html.H5("Parameters"),
                       html.P("Samples"),
                       dcc.Dropdown(
                           id="data-dropdown",
                           options=[],  # Se cambia dinamicamente
                           value=None  # Se cambia dinamicamente
                       ),
                       html.P("Taxonomic level"),
                       dcc.Dropdown(
                           id="taxo-dropdown",
                           options=[
                               {'label': level, 'value': level} 
                               for level in ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
                           ],
                           value='Phylum'  # seleccion default
                       ),
                       html.P("Type of graph"),
                       dcc.Dropdown(
                           id="graph-type-dropdown",
                           options=[
                               {'label': 'Bar Chart', 'value': 'bar'},
                               {'label': 'Heatmap', 'value': 'heatmap'},
                               {'label': 'Sankey Diagram', 'value': 'sankey'},
                               {'label': 'Sunburst Chart', 'value': 'sunburst'}
                           ],
                           value='bar'  # seleccion default
                       ),
                       html.Div(id="color-palette-container", children=[
                           html.P("Color Palette"),
                           dcc.Dropdown(
                               id="color-palette-dropdown",
                               options=[{'label': color, 'value': color} for color in px.colors.named_colorscales()],
                               value='viridis'  # seleccion default
                           ),
                       ]),
                       html.Div(id="discrete-color-container", style={'display': 'none'}, children=[
                           html.P("Color Palette"),
                           dcc.Dropdown(
                               id="discrete-color-dropdown",
                               options=[
                                   {'label': 'Set1', 'value': 'Set1'},
                                   {'label': 'Set2', 'value': 'Set2'},
                                   {'label': 'Set3', 'value': 'Set3'},
                                   {'label': 'Pastel1', 'value': 'Pastel1'},
                                   {'label': 'Pastel2', 'value': 'Pastel2'},
                                   {'label': 'Dark2', 'value': 'Dark2'},
                               ],
                               value='Set1'  # seleccion default
                           ),
                       ]),
                       html.Div(id="sample-comparison-container", style={'display': 'none'}, children=[
                            html.P("Select samples to compare:"),
                            dcc.Dropdown(
                                id="sample-comparison-dropdown",
                                options=[],
                                multi=True,
                                placeholder="Select samples to compare"
                            ),
                        ]),
                       dbc.Row([
                           dbc.Col([
                               html.P("Minimum relative abundance"),
                               dcc.Input(
                                   id="min-abundance-input",
                                   type="number",
                                   placeholder="Minimum",
                                   min=0,
                                   max=100,
                                   step=0.01,
                                   value=0
                               ),
                           ], width=4),
                           dbc.Col([
                               html.P("Maximum relative abundance"),
                               dcc.Input(
                                   id="max-abundance-input",
                                   type="number",
                                   placeholder="Maximum",
                                   min=0,
                                   max=100,
                                   step=0.1,
                                   value=100
                               ),
                           ], width=4),
                           dbc.Col([
                               html.P("Exclude taxonomic groups"),
                               dcc.Dropdown(
                                   id="exclude-taxa-dropdown",
                                   options=[],  # se cambia dinamicamente
                                   multi=True,
                                   placeholder="Select groups to exclude"
                               ),
                           ], width=4),
                       ]),
                   ]),
               ]),
               dcc.Graph(id="bar-chart"),
               dcc.Graph(id="stacked-bar-chart")
           ])
       ]),
       dcc.Tab(label='Data Table', children=[
           html.Div([
               html.H5("Data Table"),
               dcc.Dropdown(
                   id='table-sample-dropdown',
                   options=[],  # se cambia dinamicamente
                   value=None,
                   style={'width': '50%', 'marginBottom': '10px'}
               ),
               dcc.Input(
                   id='data-table-search',
                   type='text',
                   placeholder='Search table...',
                   style={'width': '50%', 'marginBottom': '10px'}
               ),
               dash_table.DataTable(
                   id='data-table',
                   page_size=15,
                   style_table={'overflowX': 'auto'},
                   style_cell={
                       'minWidth': '100px', 'width': '150px', 'maxWidth': '300px',
                       'overflow': 'hidden',
                       'textOverflow': 'ellipsis',
                       'whiteSpace': 'normal',
                       'textAlign': 'left',
                       'padding': '10px',
                       'fontFamily': 'Arial, sans-serif',
                       'fontSize': '14px'
                   },
                   style_header={
                       'backgroundColor': 'rgb(230, 230, 230)',
                       'fontWeight': 'bold',
                       'textAlign': 'center',
                       'border': '1px solid black'
                   },
                   style_data={
                       'border': '1px solid grey',
                       'backgroundColor': 'rgb(248, 248, 248)'
                   },
                   style_data_conditional=[
                       {
                           'if': {'row_index': 'odd'},
                           'backgroundColor': 'rgb(240, 240, 240)'
                       }
                   ]
               )
           ])
        ]),
        dcc.Tab(label='Download Files', children=[
            html.Div([
                html.H5("Files to download"),
                html.Div(id='download-buttons'),
                dcc.Download(id="download")
            ])
        ])
   ])
])

# Modificacion del layout de la app para incluir ambas paginas
app.layout = html.Div([
    dcc.Store(id='analysis-complete', storage_type='session'),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Div(id='analysis-progress', style={'display': 'none'}),
    html.Div(id='uploaded-files-monitor')
])

# Set up logging
logging.basicConfig(level=logging.INFO)

# Callback para actualizar el contenido de la pagina basado en la URL
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname'),
               Input('analysis-complete', 'data')])
def display_page(pathname, analysis_complete):
    if pathname == '/analysis' and analysis_complete and analysis_complete.get('complete'):
        return analysis_layout
    else:
        return upload_layout

# Funcion para parsear la salida de MetaPhlAn
def convert_path(file_path):
    if file_path.startswith(r'C:\mnt\c'):
        file_path = 'C:' + file_path[7:]
    elif file_path.startswith(r'\mnt\c'):
        file_path = 'C:' + file_path[6:]
    elif file_path.startswith('C:c'):
        file_path = 'C:' + file_path[3:]
    elif file_path.startswith('C:\\c\\'):
        file_path = 'C:' + file_path[4:]
    
    if not file_path.startswith('C:'):
        file_path = 'C:' + file_path
    
    if file_path.startswith('C:') and not file_path.startswith('C:\\'):
        file_path = 'C:\\' + file_path[2:]

    base_path = r"Users\david\OneDrive\Escritorio\Proyectos\Microbioma-Proyecto"
    if file_path.count(base_path) > 1:
        parts = file_path.split(base_path)
        file_path = 'C:\\' + base_path + parts[-1]

    if file_path.startswith('C:\\c\\'):
        file_path = 'C:\\' + file_path[5:]

    return os.path.normpath(file_path)

def parse_metaphlan_output(file_path):
    file_path = convert_path(file_path)
    
    df = pd.read_csv(file_path, sep='\t', skiprows=4)
    df['Sample'] = os.path.basename(os.path.dirname(file_path))
    df['Kingdom'] = df['#clade_name'].str.split('|').str[0].str.replace('k__', '')
    df['Phylum'] = df['#clade_name'].str.split('|').str[1].str.replace('p__', '')
    df['Class'] = df['#clade_name'].str.split('|').str[2].str.replace('c__', '')
    df['Order'] = df['#clade_name'].str.split('|').str[3].str.replace('o__', '')
    df['Family'] = df['#clade_name'].str.split('|').str[4].str.replace('f__', '')
    df['Genus'] = df['#clade_name'].str.split('|').str[5].str.replace('g__', '')
    df['Species'] = df['#clade_name'].str.split('|').str[6].str.replace('s__', '')
    return df

@app.callback(
    [Output('upload-status', 'children'),
     Output('uploaded-files-list', 'children')],
    [Input('dash-uploader', 'isCompleted')],
    [State('dash-uploader', 'fileNames'),
     State('dash-uploader', 'upload_id')]
)
def update_upload_status(is_completed, filenames, upload_id):
    logging.info(f"update_upload_status called. is_completed: {is_completed}, filenames: {filenames}")
    
    if is_completed and filenames is not None:
        for filename in filenames:
            original_file_path = os.path.join('uploads', upload_id, filename)
            folder_name = os.path.splitext(filename)[0]
            new_folder_path = os.path.join('data', folder_name)
            os.makedirs(new_folder_path, exist_ok=True)
            new_file_path = os.path.join(new_folder_path, filename)
            os.rename(original_file_path, new_file_path)
            if new_file_path not in app.uploaded_files:
                app.uploaded_files.append(new_file_path)
                logging.info(f"Added new file to app.uploaded_files: {new_file_path}")
            else:
                logging.info(f"File already in app.uploaded_files: {new_file_path}")
        
        logging.info(f"Current app.uploaded_files: {app.uploaded_files}")
        
        status = f"Files uploaded successfully: {', '.join(filenames)}. Click 'Start Analysis' to begin processing."
        file_list = html.Ul([html.Li(os.path.basename(file)) for file in app.uploaded_files])
        return status, file_list
    elif not app.uploaded_files:
        logging.info("No files in app.uploaded_files, raising PreventUpdate")
        raise PreventUpdate
    else:
        logging.info(f"No new files uploaded. Current app.uploaded_files: {app.uploaded_files}")
        file_list = html.Ul([html.Li(os.path.basename(file)) for file in app.uploaded_files])
        return "No new files uploaded.", file_list

@app.callback(
    [Output('analysis-status', 'children'),
     Output('analysis-complete', 'data'),
     Output('loading-output', 'children'),
     Output('url', 'pathname')],
    [Input('start-analysis-button', 'n_clicks')],
    prevent_initial_call=True
)
def start_analysis(n_clicks):
    if n_clicks > 0 and app.uploaded_files:
        analysis_results = []
        analysis_errors = []
        for file_path in app.uploaded_files:
            output_file, error_message = run_metaphlan_analysis(file_path)
            if output_file:
                analysis_results.append(output_file)
            else:
                analysis_errors.append(f"Error analyzing {os.path.basename(file_path)}: {error_message}")
        
        if len(analysis_results) == len(app.uploaded_files):
            analysis_status = f"Analysis completed for all {len(app.uploaded_files)} files. Results saved."
            return analysis_status, {'complete': True, 'results': analysis_results}, '', '/analysis'
        elif len(analysis_results) > 0:
            analysis_status = f"Analysis completed for {len(analysis_results)} out of {len(app.uploaded_files)} files. {len(analysis_errors)} file(s) failed: {'; '.join(analysis_errors)}"
            return analysis_status, {'complete': False}, '', '/'
        else:
            analysis_status = f"Analysis failed for all files: {'; '.join(analysis_errors)}"
            return analysis_status, {'complete': False}, '', '/'
    
    return "Click 'Start Analysis' to begin processing.", {'complete': False}, '', '/'

# Callback para actualizar las opciones de los dropdown
@app.callback(
    [Output('data-dropdown', 'options'),
     Output('data-dropdown', 'value'),
     Output('table-sample-dropdown', 'options'),
     Output('table-sample-dropdown', 'value'),
     Output('exclude-taxa-dropdown', 'options'),
     Output('sample-comparison-dropdown', 'options')],
    [Input('analysis-complete', 'data'),
     Input('taxo-dropdown', 'value')],
    [State('data-dropdown', 'value')]
)
def update_dropdown_options(analysis_data, selected_taxo, current_sample):
    if not analysis_data or not analysis_data.get('complete'):
        raise PreventUpdate
    
    results = analysis_data['results']
    sample_names = [os.path.basename(os.path.dirname(result)) for result in results]
    
    options = [{'label': name, 'value': i} for i, name in enumerate(sample_names)]
    options.append({'label': 'All', 'value': 'all-samples'})
    
    # Obtener los taxa unicos para el nivel taxonomico seleccionado
    dfs = [parse_metaphlan_output(result) for result in results]
    combined_df = pd.concat(dfs)
    unique_taxa = combined_df[selected_taxo].dropna().unique()
    exclude_options = [{'label': taxon, 'value': taxon} for taxon in unique_taxa if taxon]
    
    # Opciones para el dropdown de comparacion de muestras
    comparison_options = [{'label': name, 'value': i} for i, name in enumerate(sample_names)]
    
    # Mantener la seleccion de muestra actual
    if current_sample is None:
        current_sample = 0
    
    return options, current_sample, options, current_sample, exclude_options, comparison_options

# Callback para actualizar los graficos
@app.callback(
    [Output("bar-chart", "figure"),
     Output("stacked-bar-chart", "figure"),
     Output("stacked-bar-chart", "style")],
    [Input("taxo-dropdown", "value"),
     Input("data-dropdown", "value"),
     Input("graph-type-dropdown", "value"),
     Input("color-palette-dropdown", "value"),
     Input("discrete-color-dropdown", "value"),
     Input("min-abundance-input", "value"),
     Input("max-abundance-input", "value"),
     Input("exclude-taxa-dropdown", "value"),
     Input("sample-comparison-dropdown", "value")],
    [State('analysis-complete', 'data')]
)
def update_graph(selected_taxo, selected_data, graph_type, color_palette, discrete_color_palette, 
                 min_abundance, max_abundance, excluded_taxa, comparison_samples, analysis_data):
    if not analysis_data or not analysis_data.get('complete'):
        raise PreventUpdate

    try:
        results = analysis_data['results']
        dfs = [parse_metaphlan_output(result) for result in results]
        
        if selected_data == 'all-samples':
            if comparison_samples:
                df = pd.concat([dfs[i] for i in comparison_samples])
            else:
                df = pd.concat(dfs)
        elif selected_data is not None:
            df = dfs[int(selected_data)]
        else:
            df = dfs[0]  # Default a la primera muestra si no se selecciona ninguna
        
        # Aplicar filtros y normalizacion de datos
        df['relative_abundance'] = df['relative_abundance'].astype(float)
        total_abundance = df['relative_abundance'].sum()
        df['relative_abundance'] = (df['relative_abundance'] / total_abundance) * 100
        df = df[(df['relative_abundance'] >= min_abundance) & (df['relative_abundance'] <= max_abundance)]
        if excluded_taxa:
            df = df[~df[selected_taxo].isin(excluded_taxa)]
        
        if df.empty:
            return go.Figure(), go.Figure(), {'display': 'none'}
        
        if color_palette is None:
            color_palette = 'viridis'
        
        color_sequence = pc.sample_colorscale(color_palette, 10)
        
        if graph_type == 'bar':
            if selected_data == 'all-samples':
                sample_names = [dfs[i]['Sample'].iloc[0] for i in (comparison_samples or range(len(dfs)))]
                taxo_data = df.groupby([selected_taxo, 'Sample'])['relative_abundance'].sum().reset_index()
                taxo_data = taxo_data.pivot(index=selected_taxo, columns='Sample', values='relative_abundance').reset_index()
                taxo_data = taxo_data.sort_values(by=sample_names[0], ascending=False)
                
                if len(taxo_data) > 8:
                    fig = go.Figure()
                    for i, sample in enumerate(sample_names):
                        fig.add_trace(go.Bar(
                            y=taxo_data[selected_taxo],
                            x=taxo_data[sample],
                            name=sample,
                            orientation='h',
                            marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                        ))
                    fig.update_layout(barmode='group', xaxis_title='Relative Abundance (%)', yaxis_title=selected_taxo)
                else:
                    fig = go.Figure()
                    for i, sample in enumerate(sample_names):
                        fig.add_trace(go.Bar(
                            x=taxo_data[selected_taxo],
                            y=taxo_data[sample],
                            name=sample,
                            marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                        ))
                    fig.update_layout(barmode='group', xaxis_title=selected_taxo, yaxis_title='Relative Abundance (%)')
                
                fig.update_layout(
                    title=f'Comparison of {selected_taxo} Relative Abundance across Samples',
                    height=600,
                    width=1000
                )

                # Crear grafico stacked bar chart
                stacked_df = df.groupby([selected_taxo, 'Sample'])['relative_abundance'].sum().reset_index()
                stacked_df = stacked_df.sort_values('relative_abundance', ascending=False)
                
                # Limite de 15 taxas para mejor legibilidad
                top_taxa = stacked_df.groupby(selected_taxo)['relative_abundance'].sum().nlargest(15).index
                stacked_df = stacked_df[stacked_df[selected_taxo].isin(top_taxa)]
                
                stacked_fig = px.bar(stacked_df, 
                                     x='relative_abundance', 
                                     y=selected_taxo, 
                                     color='Sample', 
                                     orientation='h',
                                     barmode='stack',
                                     labels={'relative_abundance': 'Relative Abundance (%)'},
                                     title=f'Stacked Relative Abundance of Top 15 {selected_taxo} Across Samples')
                
                stacked_fig.update_layout(height=600, width=1000, yaxis={'categoryorder':'total ascending'})
                
                return fig, stacked_fig, {'display': 'block'}
            else:
                taxo_data = df.groupby(selected_taxo)['relative_abundance'].sum().reset_index().sort_values(by='relative_abundance', ascending=False)
                if len(taxo_data) > 8:
                    fig = px.bar(taxo_data, y=selected_taxo, x='relative_abundance', color=selected_taxo, color_discrete_sequence=color_sequence, orientation='h')
                    fig.update_xaxes(range=[0, 100])
                else:
                    fig = px.bar(taxo_data, x=selected_taxo, y='relative_abundance', color=selected_taxo, color_discrete_sequence=color_sequence)
                    fig.update_yaxes(range=[0, 100])
                fig.update_layout(title=f'Relative Abundance of {selected_taxo} in {"All" if selected_data == "all-samples" else df["Sample"].iloc[0]} Sample', height=600, width=1000)
                return fig, go.Figure(), {'display': 'none'}
        elif graph_type == 'heatmap':
            heatmap_data = df.pivot_table(index=selected_taxo, columns='Sample', values='relative_abundance', aggfunc='sum').fillna(0)
            
            # normaliza los datos del heatmap para asegurar que cada sample sume 100%
            heatmap_data = heatmap_data.apply(lambda x: (x / x.sum()) * 100, axis=0)
            
            custom_colorscale = [
                [0, "rgb(242,240,247)"],    # morado ligero
                [0.0002, "rgb(234,231,240)"],
                [0.0004, "rgb(226,222,234)"],
                [0.0006, "rgb(218,213,228)"],
                [0.0008, "rgb(210,204,222)"],
                [0.001, "rgb(202,195,216)"],
                [0.002, "rgb(194,186,210)"],
                [0.004, "rgb(186,177,204)"],
                [0.006, "rgb(178,168,198)"],
                [0.008, "rgb(170,159,192)"],
                [0.01, "rgb(162,150,186)"],
                [0.02, "rgb(154,141,180)"],
                [0.04, "rgb(146,132,174)"],
                [0.06, "rgb(138,123,168)"],
                [0.08, "rgb(130,114,162)"],
                [0.1, "rgb(122,105,156)"],
                [0.2, "rgb(114,96,150)"],
                [0.4, "rgb(106,87,144)"],
                [0.6, "rgb(98,78,138)"],
                [0.8, "rgb(90,69,132)"],
                [1, "rgb(82,60,126)"]       # morado oscuro
            ]            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale=custom_colorscale,
                colorbar=dict(title='Relative Abundance', ticklen=0, ticktext=['0%', '10%', '20%', '30%', '40%', '50%'], tickvals=[0, 10, 20, 30, 40, 50]),
                zmin=0.001, zmax=50
            ))
            fig.update_layout(
                title=f'Heatmap of {selected_taxo} Relative Abundance',
                xaxis_title='Sample',
                yaxis_title=selected_taxo,
                height=600,
                width=1000
            )
            fig.update_traces(
                hovertemplate='Sample: %{x}<br>' + selected_taxo + ': %{y}<br>Relative Abundance: %{z:.2f}%<extra></extra>'
            )
            return fig, go.Figure(), {'display': 'none'}
        elif graph_type == 'sankey':
            fig = create_sankey_diagram(df, selected_taxo, selected_data)
            return fig, go.Figure(), {'display': 'none'}
        elif graph_type == 'sunburst':
            fig = create_sunburst_chart(df, selected_taxo)
            return fig, go.Figure(), {'display': 'none'}
        else:
            return go.Figure(), go.Figure(), {'display': 'none'}
    except Exception as e:
        print(f"Error in update_graph: {e}")
        return go.Figure(), go.Figure(), {'display': 'none'}

# Funciones para crear los diagramas de Sankey y sunburst_chart
def create_sankey_diagram(df, selected_taxo, selected_data):
    # Define los niveles taxonomicos en orden
    tax_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    # Para encontrar el indice del nivel taxonomico seleccionado
    selected_index = tax_levels.index(selected_taxo)
    
    # Usar solo los niveles hasta e incluyendo el nivel seleccionado
    used_levels = tax_levels[:selected_index+1]
    
    # Preparar los datos para el diagrama de Sankey
    nodes = []
    links = []
    node_dict = {}

    # Crear un color palette
    colors = px.colors.qualitative.Plotly[:len(used_levels)]
    color_dict = {level: colors[i] for i, level in enumerate(used_levels)}

    def adjust_color(hex_color, factor):
        rgb = pc.hex_to_rgb(hex_color)
        adjusted_rgb = [min(255, max(0, int(c * factor))) for c in rgb]
        return f'rgb({adjusted_rgb[0]}, {adjusted_rgb[1]}, {adjusted_rgb[2]})'

    for i, level in enumerate(used_levels):
        level_data = df.groupby(level)['relative_abundance'].sum().reset_index()
        level_color = color_dict[level]
        
        for _, row in level_data.iterrows():
            if row[level] not in node_dict:
                node_dict[row[level]] = len(nodes)
                color_factor = 0.5 + 0.5 * (len(node_dict) / len(level_data))
                adjusted_color = adjust_color(level_color, color_factor)
                nodes.append(dict(label=row[level], color=adjusted_color))
            
            if i > 0:
                parent_level = used_levels[i-1]
                parent = df[df[level] == row[level]][parent_level].iloc[0]
                source = node_dict[parent]
                target = node_dict[row[level]]
                value = row['relative_abundance']
                links.append(dict(source=source, target=target, value=value))

    # Crear el diagrama de Sankey
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = [node['label'] for node in nodes],
          color = [node['color'] for node in nodes]
        ),
        link = dict(
          source = [link['source'] for link in links],
          target = [link['target'] for link in links],
          value = [link['value'] for link in links]
      ))])

    fig.update_layout(title_text=f"Taxonomic Hierarchy up to {selected_taxo}", font_size=10)
    return fig

def create_sunburst_chart(df, selected_taxo):
    try:
        tax_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        selected_index = tax_levels.index(selected_taxo)
        used_levels = tax_levels[:selected_index+1]
        
        df_sunburst = df[used_levels + ['relative_abundance']].copy()
        
        # Rellena los valores NaN con un placeholder
        for level in used_levels:
            df_sunburst[level] = df_sunburst[level].fillna(f"Unknown {level}")
        
        # normaliza relative abundance para asegurar que suma 100%
        total_abundance = df_sunburst['relative_abundance'].sum()
        df_sunburst['relative_abundance'] = (df_sunburst['relative_abundance'] / total_abundance) * 100
        
        # Crear etiquetas y parents
        df_sunburst['labels'] = df_sunburst[used_levels].apply(lambda row: ' | '.join(row), axis=1)
        df_sunburst['parents'] = df_sunburst[used_levels].apply(lambda row: ' | '.join(row[:-1]), axis=1)
        df_sunburst.loc[df_sunburst['parents'] == '', 'parents'] = ''
        
        fig = px.sunburst(
            df_sunburst,
            path=used_levels,
            values='relative_abundance',
            hover_data=['relative_abundance']
        )
        
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Abundance: %{value:.2f}%<br>Parent: %{parent}'
        )
        
        fig.update_layout(
            title=f'Sunburst Chart of Taxonomic Hierarchy up to {selected_taxo}',
            margin=dict(t=30, l=0, r=0, b=0)
        )
        
        return fig
    except Exception as e:
        print(f"Error in create_sunburst_chart: {e}")
        return go.Figure()  # Muestra un grafico vacio en caso de error

# Callback para la tabla de datos
@app.callback(
    [Output('data-table', 'data'),
     Output('data-table', 'columns')],
    [Input('table-sample-dropdown', 'value'),
     Input('data-table-search', 'value')],
    [State('analysis-complete', 'data')]
)
def update_table(selected_sample, search_value, analysis_data):
    if not analysis_data or not analysis_data.get('complete'):
        raise PreventUpdate
    
    results = analysis_data['results']
    dfs = [parse_metaphlan_output(result) for result in results]
    
    if selected_sample == 'all-samples':
        df = pd.concat(dfs)
    else:
        df = dfs[int(selected_sample)]
    
    columns = [{"name": i, "id": i} for i in df.columns]
    
    if search_value:
        df = df[df.apply(lambda row: any(search_value.lower() in str(cell).lower() for cell in row), axis=1)]
    
    data = df.to_dict('records')
    return data, columns

# Callback para los botones de descarga
@app.callback(
    Output('download-buttons', 'children'),
    [Input('analysis-complete', 'data')]
)
def update_download_buttons(analysis_data):
    if not analysis_data or not analysis_data.get('complete'):
        return []
    
    results = analysis_data['results']
    buttons = []
    
    for i, result in enumerate(results):
        sample_name = os.path.basename(os.path.dirname(result))
        buttons.extend([
            html.Button(f"Download {sample_name}.bz2", id={'type': 'download-button', 'index': f'{i}-bz2'}),
            html.Button(f"Download {sample_name}.csv", id={'type': 'download-button', 'index': f'{i}-csv'})
        ])
    
    buttons.append(html.Button("Download Combined Data Table", id="download-combined-table-button"))
    
    return buttons

@app.callback(
    Output("download", "data"),
    [Input({'type': 'download-button', 'index': ALL}, 'n_clicks'),
     Input("download-combined-table-button", "n_clicks")],
    [State('analysis-complete', 'data')],
    prevent_initial_call=True
)
def download_files(n_clicks, n_clicks_combined, analysis_data):
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]['value'] is None:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "download-combined-table-button":
        if not analysis_data or not analysis_data.get('complete'):
            raise PreventUpdate
        results = analysis_data['results']
        dfs = [parse_metaphlan_output(result) for result in results]
        combined_df = pd.concat(dfs)
        return dcc.send_data_frame(combined_df.to_csv, "combined_data_table.csv", index=False)
    
    file_index, file_type = json.loads(button_id)['index'].split('-')
    file_index = int(file_index)
    
    if file_type == 'bz2':
        # encuentra el archivo original .bz2
        sample_folder = os.path.dirname(analysis_data['results'][file_index])
        bz2_files = [f for f in os.listdir(sample_folder) if f.endswith('.bz2')]
        if not bz2_files:
            raise PreventUpdate
        bz2_file_path = os.path.join(sample_folder, bz2_files[0])
        return dcc.send_file(bz2_file_path)
    elif file_type == 'csv':
        # genera un archivo csv con los resultados
        df = parse_metaphlan_output(analysis_data['results'][file_index])
        sample_name = os.path.basename(os.path.dirname(analysis_data['results'][file_index]))
        return dcc.send_data_frame(df.to_csv, f"{sample_name}_results.csv", index=False)

    raise PreventUpdate

@app.callback(
    Output('exclude-taxa-dropdown', 'value'),
    [Input('taxo-dropdown', 'value')]
)
def clear_excluded_taxa(selected_taxo):
    return []

@app.callback(
    Output('uploaded-files-monitor', 'children'),
    Input('interval-component', 'n_intervals')
)
def monitor_uploaded_files(n):
    return f"Current uploaded files: {app.uploaded_files}"

# componente de intervalo para actualizar la pagina cada 1000 ms
dcc.Interval(id='interval-component', interval=1000, n_intervals=0)

@app.callback(
    Output('sample-comparison-container', 'style'),
    [Input('data-dropdown', 'value')]
)
def toggle_sample_comparison(selected_data):
    if selected_data == 'all-samples':
        return {'display': 'block'}
    return {'display': 'none'}

if __name__ == '__main__':
    app.run_server(debug=True)