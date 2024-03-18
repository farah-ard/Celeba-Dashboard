from dash import Dash, html, dcc, callback, Output, Input, State
import dash
import dash_bootstrap_components as dbc

import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from PIL import Image

#Creating the app
app = Dash(external_stylesheets=[dbc.themes.PULSE])
app.title = 'CelebA-Vis'

#Opening the datasets and computing metrics
dataset_names = ["celeba_buffalo_s.csv", "celeba_buffalo_l.csv"]
data_frames = [pd.read_csv("celeba/"+dataset_names[0]), pd.read_csv("celeba/"+dataset_names[1])]

default_df = data_frames[0]

embeddings = [var for var in default_df.columns if var.startswith('embedding')]

ids = default_df['id'].values.tolist()
attributes = [x for x in default_df.columns if x not in embeddings]
attributes.remove('image_name')
attributes.remove('id')

default_num_rows = [len(data_frames[0]), len(data_frames[1])]
default_num_ids = [len(data_frames[0]['id'].unique()), len(data_frames[1]['id'].unique())]
default_num_na_values = [data_frames[0].isna().sum().sum(), data_frames[1].isna().sum().sum()]
default_num_embeddings = [len(embeddings), len(embeddings)]

#Setting values for the dropdown menus
cluster_methods = ['K-means', 'DBSCAN', 'Hierarchical']
dr_methods = ['PCA', 'UMAP']

n_clusters = [1,2,3,4,5,6,7,8,9,10]
n_neighbours = [5, 10, 20, 50]
min_dist = [0.2,0.4,0.6,0.8]

#Sidebar layout
sidebar = html.Div(
    [
        dbc.Row(
            [
                html.P('Main Menu')
                ],
            style={"height": "5vh"}, className='bg-primary text-white'
            ),
        dbc.Row(
            [
                html.P('Dataset Selection',
                           style={'margin-top': '8px', 'margin-bottom': '4px'},
                           className='text-white'),
                    dcc.Dropdown(id='dataset-picker', multi=False, value=dataset_names[0],
                                 options=[{'label': x, 'value': x}
                                          for x in dataset_names],
                                 style={'width': '100%'}
                                 ),
                ],
            style={"height": "15vh"}, className='bg-secondary text-body-emphasis'
            ),
        dbc.Row(
            [
                html.P('Metrics'),
                html.Div(id='metrics-output')
            ],
            style={"height": "40vh"}, className='bg-secondary text-white'
        ),

        dbc.Row(
            [
                html.Div([
                        html.P('Person ID', style={'margin-top': '8px', 'margin-bottom': '4px'}, className='text-white'),
                        dcc.Dropdown(id='id-picker', multi=False, value=None,
                                    options=[{'label': x, 'value': x} for x in ids],
                                    style={'width': '100%'}
                                    ),
                        html.P('Physical Attributes', style={'margin-top': '16px', 'margin-bottom': '4px'}, className='text-white'),
                        dcc.Dropdown(id='attribute-picker', multi=True, value=[attributes[1]],
                                    options=[{'label': x, 'value': x} for x in attributes],
                                    style={'width': '100%'}
                                    ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.P('Clustering Method',
                                            style={'margin-top': '16px', 'margin-bottom': '4px'},
                                            className='text-white'),
                                        dcc.Dropdown(id='cluster-picker', multi=False, value=cluster_methods[0],
                                                    options=[{'label': x, 'value': x} for x in cluster_methods],
                                                    style={'width': '100%'}
                                                    ),
                                    ],
                                ),
                                dbc.Col(
                                    [
                                        html.P('DR Method',
                                            style={'margin-top': '16px', 'margin-bottom': '4px'},
                                            className='text-white'),
                                        dcc.Dropdown(id='dr-picker', multi=False, value=dr_methods[0],
                                                    options=[{'label': x, 'value': x} for x in dr_methods],
                                                    style={'width': '100%'}
                                                    ),
                                    ],
                                )
                            ],
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.P('Clusters',
                                            style={'margin-top': '16px', 'margin-bottom': '4px'},
                                            className='text-white'),
                                        dcc.Dropdown(id='n-clusters-picker', multi=False, value=n_clusters[9],
                                                    options=[{'label': x, 'value': x} for x in n_clusters],
                                                    style={'width': '100%'}
                                                    ),
                                    ],
                                ),
                                dbc.Col(
                                    [
                                        html.P('Neighbours',
                                            style={'margin-top': '16px', 'margin-bottom': '4px'},
                                            className='text-white'),
                                        dcc.Dropdown(id='neighbours-picker', multi=False, value=n_neighbours[0],
                                                    options=[{'label': x, 'value': x} for x in n_neighbours],
                                                    style={'width': '100%'}
                                                    ),
                                    ],
                                ),
                                dbc.Col(
                                    [
                                        html.P('Distance',
                                            style={'margin-top': '16px', 'margin-bottom': '4px'},
                                            className='text-white'),
                                        dcc.Dropdown(id='distance-picker', multi=False, value=min_dist[0],
                                                    options=[{'label': x, 'value': x} for x in min_dist],
                                                    style={'width': '100%'}
                                                    ),
                                    ],
                                )
                            ],
                            style={'height': '40vh'}, className='bg-secondary text-body-emphasis'
                        )
                    ])

                ],
            style={'height': '40vh'}, className='bg-secondary text-body-emphasis'
            )
        ]
    )

#Plot zone layout
content = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div([
                            dcc.Graph(id="pie-chart", className='bg-light', style={'position': 'absolute','height': '100%', 'width':'100%' }),
                            dbc.Button("Next Attribute", id='next-attribute-button', n_clicks=0, color="primary",
                                       style={'position': 'absolute', 'bottom': '0', 'right': '0', 'margin': '10px'})
                        ]),
                    ],
                    style={'position': 'relative', 'height': '50vh', 'width': '50%'}
                ),
                dbc.Col(
                    [
                        dcc.Graph(id='corr-plot', style={'height': '100%', 'width':'100%' })
                    ],
                    className='bg-light', style={'position': 'relative', 'height': '50vh', 'width': '50%'}
                )
            ],
            style={"height": "50vh",'margin-top': '16px', 'margin-left': '8px',
                   'margin-bottom': '8px', 'margin-right': '8px'}
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(id='cluster-plot', style={'height': '100%', 'width':'100%' }),
                    ],
                    className='bg-light', style={'position': 'relative', 'height': '70vh', 'width':'70%' }
                ),
                dbc.Col(
                    [
                        dcc.Graph(id='image-display', style={'height': '100%', 'width':'100%' })  
                    ],
                    className='bg-light', style={'position': 'relative', 'height': '70vh', 'width':'30%' }
                )
            ],
            style={"height": "70vh", 'margin-top': '16px', 'margin-left': '8px',
                   'margin-bottom': '8px', 'margin-right': '8px'}
        )
    ]
)

#App layout combining sidebar and content
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=3, className='bg-light'),
                dbc.Col(content, width=9, className='bg-light')
                ],
            style={"height": "120vh"}
            )
        ],
    fluid=True
    )

@app.callback(
    Output('metrics-output', 'children'),
    [
        Input('dataset-picker', 'value'),
    ]
)
#Loads and displays the metrics when the dataset is changed
def update_metrics(selected_dataset):
    if selected_dataset == dataset_names[0]:
        num_rows = default_num_rows[0]
        num_ids = default_num_ids[0]
        num_na_values = default_num_na_values[0]
        num_embeddings = default_num_embeddings[0]
    else:
        num_rows = default_num_rows[1]
        num_ids = default_num_ids[1]
        num_na_values = default_num_na_values[1]
        num_embeddings = default_num_embeddings[1]

    # Create HTML content for the metrics
    metrics_card = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.P(f'Number of Rows: {num_rows}', className="card-text"),
                    html.P(f'Number of IDs: {num_ids}', className="card-text"),
                    html.P(f'Number of NA Values: {num_na_values}', className="card-text"),
                    html.P(f'Number of Embeddings: {num_embeddings}', className="card-text"),
                ]
            )
        ]
    )

    return metrics_card

@app.callback(
    
    Output('pie-chart', 'figure'),
    [
        Input('dataset-picker', 'value'),
        Input('id-picker', 'value'),
        Input('attribute-picker', 'value'),
        Input('next-attribute-button', 'n_clicks'),
    ]
)
#Updates pie chart view when an attribute is picked, next attribute button is clicked, ID is picked or dataset is modified
def update_pie_chart(selected_dataset, selected_id, selected_attributes, n_clicks):
    #Loads the selected dataset
    if selected_dataset == dataset_names[0]:
        df = data_frames[0]
    else:
        df = data_frames[1]

    #Filters data based on the selected person ID and attributes
    if selected_id == None:
        filtered_df = df[selected_attributes] 
    else:
        filtered_df = df[df['id'] == selected_id][selected_attributes]  
    
    #Computes proportions
    nb_rows = filtered_df.shape[0]

    current_attribute_index = n_clicks % len(selected_attributes)
    column = selected_attributes[current_attribute_index]
    labels = [column, 'Not '+column]
    sizes = [nb_rows*np.mean(filtered_df[column] == 1), nb_rows*(1-np.mean(filtered_df[column] == 1))]

    #Creates a pie chart
    fig = px.pie(df, values=sizes, names=labels, color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(
        title=f"Pie Chart for {selected_id}'s {column}",
    )
    
    return fig



@app.callback(
     Output('corr-plot', 'figure'),
    [
        Input('dataset-picker', 'value'),
        Input('attribute-picker', 'value'),
    ]
)
#Updates correlation plot view when an attribute is picked or dataset is modified
def update_corr_plot(selected_dataset, selected_attributes):
     #Loads the selected dataset
    if selected_dataset == dataset_names[0]:
        df = data_frames[0]
    else:
        df = data_frames[1]
    #Computes the correlation matrix and plot
    corr_matrix = df[selected_attributes].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=selected_attributes,
        y=selected_attributes,
        colorscale=plotly.colors.sequential.Bluyl))
    fig.update_layout( 
        title='Correlation Between Attributes')
    return fig
    

@app.callback(
    Output('cluster-plot', 'figure'),
    [
        Input('dataset-picker', 'value'),
        Input('cluster-picker', 'value'),
        Input('dr-picker', 'value'),
        Input('attribute-picker', 'value'),
        Input('n-clusters-picker', 'value'),
        Input('distance-picker', 'value'),
        Input('neighbours-picker', 'value'),
    ]
)
#Updates cluster plot view when input is changed
def update_clusters(selected_dataset, selected_clustering, selected_dr, selected_attributes, n_clusters, dist, n_neighbours):
    if selected_dataset == dataset_names[0]:
        df = data_frames[0]
    else:
        df = data_frames[1]

    if selected_clustering == 'Hierarchical' and len(selected_attributes)>2:
        for att in selected_attributes:
            df = df[df[att] == 1]
        fig = ff.create_dendrogram(df[embeddings], labels = df['image_name'].tolist())

    else:
        #Standarizes the data
        scaler = StandardScaler()
        df_std = scaler.fit_transform(df[embeddings])

        if selected_dr == 'PCA':
            if selected_dataset == dataset_names[0]:
            #Applies PCA for the first reduction
                pca_first = PCA(n_components=255) 
            else:
                pca_first = PCA(n_components=311) 
            proj = pca_first.fit_transform(df_std)

            #Applies clustering on the reduced data
            if selected_clustering == 'K-means':
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
                df['cluster'] = kmeans.fit_predict(proj)
            if selected_clustering == 'DBSCAN':
                dbscan = DBSCAN(eps=dist,min_samples=n_neighbours)
                df['cluster'] = dbscan.fit_predict(proj)

            
            #Applies PCA with 2 components
            pca_second = PCA(n_components=2)
            scores_pca_second = pca_second.fit_transform(df.drop('cluster', axis=1)[embeddings])

            #Creates a DataFrame for plotting
            plot_df = pd.DataFrame({
                'PC1': scores_pca_second[:, 0],
                'PC2': scores_pca_second[:, 1],
                'Cluster': df['cluster'],
                'image_name': df['image_name'],
                'id': df['id']
            })

            #Plots
            fig = px.scatter(
                plot_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                title='Clusters Visualized in PCA Space',
                labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                color_continuous_scale=px.colors.qualitative.Set3,
                hover_data={'image_name': True, 'id':True},
            )
        else:
            #Applies UMAP reduction
            umap_2d = UMAP(n_components=2, n_neighbors = n_neighbours, min_dist= dist, metric='euclidean')
            proj = umap_2d.fit_transform(df_std)

             #Applies clustering on the reduced data
            if selected_clustering == 'K-means':
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
                df['cluster'] = kmeans.fit_predict(proj)
            if selected_clustering == 'DBSCAN':
                dbscan = DBSCAN(eps=dist,min_samples=n_neighbours, metric='euclidean')
                df['cluster'] = dbscan.fit_predict(proj)
            #Creates a DataFrame for plotting
            plot_df = pd.DataFrame({
                'PC1': proj[:, 0],
                'PC2': proj[:, 1],
                'Cluster': df['cluster'],
                'image_name': df['image_name'],
                'id': df['id']
            })
            #Plots
            fig = px.scatter(
                plot_df, x='PC1', y='PC2',
                color='Cluster',
                title='Clusters Visualized in UMAP Space',
                labels={'PC1': 'UMAP Component 1', 'PC2': 'UMAP Component 2'},
                color_continuous_scale=px.colors.qualitative.Set3,
                hover_data={'image_name': True, 'id':True},
            )

   
    return fig

@app.callback(
    Output('image-display', 'figure'), 
    [   
        Input('dataset-picker', 'value'),
        Input('cluster-plot', 'clickData'),
        Input('id-picker', 'value'),
    ] 
)
#Displays image
def display_selected_image(selected_data, clickData,pickedID):
    image_path = "celeba/img_celeba/000001.jpg"
    if pickedID is not None:
        if selected_data == dataset_names[0]:
            df = data_frames[0]
        else:
            df = data_frames[1]
        image_name = df[df['id']==pickedID]['image_name'].iloc[0]
        image_path = f"celeba/img_celeba/{image_name}"
     
    elif clickData is not None and 'customdata' in clickData['points'][0]:
        image_name = clickData['points'][0]['customdata']
        image_path = f"celeba/img_celeba/{image_name[0]}"
    
    img = np.asarray(Image.open(image_path))
    imgplot = px.imshow(img)
    imgplot.update_xaxes(showticklabels=False)
    imgplot.update_yaxes(showticklabels=False)

    return imgplot 



if __name__ == '__main__':
    app.run_server(debug=False)

