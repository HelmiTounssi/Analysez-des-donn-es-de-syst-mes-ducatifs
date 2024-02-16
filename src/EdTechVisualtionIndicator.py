# Nom du fichier : mymodule.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os
import ipywidgets as widgets
from ipydatagrid import DataGrid
from IPython.display import display
import numpy as np
import plotly.express as px
from IPython.display import clear_output
def read_file(indicatorName):
    df = pd.read_csv('../data/indicatorCodes/' + indicatorName + '.csv')
    return df


def create_box_plot(dataframe):
    columns_1970_2007 = ['2000']
    df_filtered1 = dataframe[['Country Name'] + columns_1970_2007]
    df_sorted = df_filtered1.sort_values('2000', ascending=True)
    TOP50 = df_sorted.head(50)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Country Name', y='2000', data=TOP50)
    plt.xticks(rotation=90)
    plt.title('Variation between Countries')
    plt.xlabel('Country')
    plt.ylabel('Value')
    plt.show()


def create_heatmap(dataframe):
    columns_to_reshape = ['2000', '2005', '2006', '2007',
                          '2008', '2009', '2010', '2011', '2012', '2013']
    df_reshaped = pd.melt(dataframe, id_vars=[
                          'Country Name'], value_vars=columns_to_reshape, var_name='Year', value_name='Value')
    pivot_df = df_reshaped.pivot(
        index='Country Name', columns='Year', values='Value')

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, cmap='YlOrRd')
    plt.title('Variation between Countries')
    plt.xlabel('Variable')
    plt.ylabel('Country')
    plt.show()


def create_zoomable_heatmap(dataframe):
    columns_to_reshape = ['2000', '2005', '2006', '2007',
                          '2008', '2009', '2010', '2011', '2012', '2013']
    columns_to_keep = ['Country Name', '2000', '2005', '2006',
                       '2007', '2008', '2009', '2010', '2011', '2012', '2013']

    df_dropped = dataframe[columns_to_keep]
    df_sorted = df_dropped.sort_values('2000', ascending=False)

    fig = go.Figure(data=go.Heatmap(
        z=df_sorted.iloc[:, 1:].values,
        x=df_sorted.columns[1:],
        y=df_sorted['Country Name'],
        colorscale='Viridis',
    ))

    fig.update_layout(
        title='Zoomable Heatmap',
        xaxis=dict(title='Categories'),
        yaxis=dict(title='Countries'),
        autosize=False,
        width=800,
        height=500,
        yaxis_scaleanchor='x',
        yaxis_scaleratio=0.75,
    )

    fig.update_layout(
        dragmode='zoom',
        hovermode='closest',
    )

    fig.show()


def split_data_by_indicator_save_csv(input_column, input_filename, output_directory):
    # Lecture du fichier CSV
    df = pd.read_csv(input_filename, on_bad_lines='skip',
                     engine='python', sep=',')

    # Parcours des indicateurs uniques dans la colonne 'Indicator Code'
    for indicatorCode in df[input_column].unique():
        # Filtrage des lignes pour l'indicateur spécifique
        filtered_df = df[df[input_column] == indicatorCode]

        # Création du répertoire si nécessaire
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Enregistrement du fichier dans le répertoire par nom d'indicateur
        output_filename = os.path.join(
            output_directory, f"{indicatorCode}.csv")
        filtered_df.to_csv(output_filename, index=False)

    return df


def save_statistics_by_region(data_file, country_file,nom_fichier,repertoire_destination):
    # Lecture du premier fichier CSV
    df1 = pd.read_csv(data_file, on_bad_lines='skip', engine='python', sep=',')
    # Lecture du deuxième fichier CSV
    df_Country = pd.read_csv(
        country_file, on_bad_lines='skip', engine='python', sep=',')

    # Spécifier la colonne de jointure commune entre les deux fichiers
    colonne_jointure = 'Country Code'

    # Effectuer la jointure entre les deux DataFrames sur la colonne de jointure
    df_resultat = pd.merge(
        df1, df_Country[[colonne_jointure, 'Region']], on=colonne_jointure, how='left')

    # Liste des colonnes numériques pour lesquelles vous souhaitez calculer la moyenne
    colonnes_numeriques = ['2000', '2005', '2006', '2007',
                           '2008', '2009', '2010', '2011', '2012', '2013']

    # Calcul de la moyenne des colonnes par région
    df_resultat['moyennes'] = df_resultat[colonnes_numeriques].mean(axis=1)
    df_resultat['medians'] = df_resultat[colonnes_numeriques].median(axis=1)
    df_resultat['EcartTypes'] = df_resultat[colonnes_numeriques].std(axis=1)

    metrics_par_region_df = df_resultat[[
        'Region', 'Indicator Code', 'Indicator Name', 'moyennes', 'medians', 'EcartTypes']]

    # Calcul de la moyenne par région
    moyenne_grouped_data_par_region_df = metrics_par_region_df.groupby(['Region', 'Indicator Code', 'Indicator Name'])[
        'moyennes'].mean()
    # Renommer la colonne calculée
    moyenne_grouped_data_par_region_df = moyenne_grouped_data_par_region_df.rename(
        'Moyenne')
    # Fusionner les résultats avec le DataFrame d'origine
    xxx_par_region_df = pd.merge(metrics_par_region_df, moyenne_grouped_data_par_region_df,
                                 left_on=['Region', 'Indicator Code', 'Indicator Name'], right_index=True,
                                 how='left')

    median_grouped_data_par_region_df = metrics_par_region_df.groupby(['Region', 'Indicator Code', 'Indicator Name'])[
        'medians'].median()
    # Renommer la colonne calculée
    median_grouped_data_par_region_df = median_grouped_data_par_region_df.rename(
        'Mediane')
    # Fusionner les résultats avec le DataFrame d'origine
    xxx_par_region_df = pd.merge(xxx_par_region_df, median_grouped_data_par_region_df,
                                 left_on=['Region', 'Indicator Code', 'Indicator Name'], right_index=True,
                                 how='left')

    ecartType_grouped_data_par_region_df = metrics_par_region_df.groupby(['Region', 'Indicator Code', 'Indicator Name'])[
        'EcartTypes'].std()
    # Renommer la colonne calculée
    ecartType_grouped_data_par_region_df = ecartType_grouped_data_par_region_df.rename(
        'Ecart Type')

    # Fusionner les résultats avec le DataFrame d'origine
    # Fusionner les résultats avec le DataFrame d'origine
    # Fusionner les résultats avec le DataFrame d'origine
    xxx_par_region_df = pd.merge(xxx_par_region_df, ecartType_grouped_data_par_region_df, left_on=['Region', 'Indicator Code', 'Indicator Name'], right_index=True, how='left')


    # Supprimer les lignes avec des doublons basés sur deux colonnes identiques

    xxx_par_region_df.dropna(subset=['Region'], inplace=True)
    colonnes_selectionnees = ['Region','Indicator Code','Indicator Name','Moyenne','Mediane','Ecart Type']
    xxx_par_region_df=xxx_par_region_df.loc[:, colonnes_selectionnees]
    df_gpd = pd.read_csv('../data/statistiqueParRegion/gpdByRegion.csv',on_bad_lines='skip',engine='python', sep=',')

    #xxx_par_region_df = pd.merge(xxx_par_region_df, df_gpd, left_on=['Region'], right_index=True, how='left')
    nouveau_dataframe = pd.concat([xxx_par_region_df, df_gpd])
    nouveau_dataframe['Region'] = nouveau_dataframe['Region'].astype(str)
    nouveau_dataframe['Indicator Code'] = nouveau_dataframe['Indicator Code'].astype(str)

    xxx_par_region_df1 = nouveau_dataframe.drop_duplicates(subset=['Region', 'Indicator Code'], keep='first')



    # Chemin complet du répertoire de destination
    chemin_repertoire = os.path.join(os.getcwd(), repertoire_destination)

    # Vérifier si le répertoire existe, sinon le créer
    if not os.path.exists(chemin_repertoire):
        os.makedirs(chemin_repertoire)

    # Chemin complet du fichier de destination
    chemin_fichier = os.path.join(chemin_repertoire, nom_fichier)

    # Sauvegarder le DataFrame dans un fichier CSV
    xxx_par_region_df1.to_csv(chemin_fichier, index=False)

    # Affichage d'un message de confirmation
    #print("Le fichier a été sauvegardé avec succès dans le répertoire statistiqueParPays.")
    return xxx_par_region_df1


def create_datagrid_by_indicator(filename):
    # Lecture du premier fichier CSV
    df = pd.read_csv(filename, on_bad_lines='skip', engine='python', sep=',')

    # Extraire les noms des colonnes (indicateurs)
    indicateurs = df['Indicator Code'].unique()

    # Créer un widget ListBox avec les indicateurs
    listbox = widgets.Select(
        options=indicateurs,
        description='Indicateurs:',
        rows=10
    )
    # Créer un widget DataGrid vide
    datagrid = widgets.Output()

    # Fonction de mise à jour du DataGrid
    def update_datagrid(indicateur):
        with datagrid:
            # Supprimer le contenu précédent du DataGrid
            datagrid.clear_output()
            nouveau_df = df[df['Indicator Code'] == indicateur]
            display(nouveau_df)

    # Lier la fonction de mise à jour du DataGrid à l'événement de sélection du ListBox
    listbox.observe(lambda event: update_datagrid(event['new']), names='value')

    # Retourner le ListBox et le DataGrid dans une disposition
    return widgets.VBox([listbox, datagrid])
def create_datagrid_and_visualizer_by_indicator(filename):
    # Lecture du premier fichier CSV
    df = pd.read_csv(filename, on_bad_lines='skip', engine='python', sep=',')

    # Extraire les noms des colonnes (indicateurs)
    indicateurs = df['Indicator Code'].unique()

    # Créer un widget ListBox avec les indicateurs
    listbox = widgets.Select(
        options=indicateurs,
        description='Indicateurs:',
        rows=10
    )
    # Créer un widget DataGrid vide
    datagrid = widgets.Output()

    # Fonction de mise à jour du DataGrid
    def update_datagrid(indicateur):
        with datagrid:
            # Supprimer le contenu précédent du DataGrid
            datagrid.clear_output()
            clear_output(wait=True)
            dataframe = df[df['Indicator Code'] == indicateur]
            dataframe = read_file(indicateur)
            #create_box_plot(dataframe)
            create_heatmap(dataframe)
            create_zoomable_heatmap(dataframe)
            generate_indicator_map(indicateur)
            display(dataframe)

    # Lier la fonction de mise à jour du DataGrid à l'événement de sélection du ListBox
    listbox.observe(lambda event: update_datagrid(event['new']), names='value')

    # Retourner le ListBox et le DataGrid dans une disposition
    return widgets.VBox([listbox, datagrid])


def generate_indicator_map(indicatorCode):
    distinct_values = [indicatorCode]
    nom_fichier=indicatorCode+"_ByPays.csv"
    pd.options.mode.chained_assignment = None
    df = pd.read_csv('../data/EdStatsData.csv', on_bad_lines='skip', engine='python', sep=',')

    # Filtrer le DataFrame sur les indicateurs spécifiques
    filtered_df = df[df['Indicator Code'].isin(distinct_values)]

    # Sélectionner les colonnes pertinentes
    columns_below_mean_nulls = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '2000', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013']
    df = filtered_df[columns_below_mean_nulls]

    # Calculer les moyennes, médianes et écart-types
    colonnes_numeriques = ['2000', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013']
    df['moyennes'] = df[colonnes_numeriques].mean(axis=1)
    df['medians'] = df[colonnes_numeriques].median(axis=1)
    df['EcartTypes'] = df[colonnes_numeriques].std(axis=1)

    # Sélectionner les colonnes finales
    colonnes_selectionnees = ['Country Name', 'Indicator Code', 'Indicator Name', 'moyennes', 'medians', 'EcartTypes']
    df.drop(columns=df.columns.difference(colonnes_selectionnees), inplace=True)

    # Calcul des percentiles et attribution des catégories
    column_name = 'moyennes'
    if column_name in df.columns:
        percentiles = df[column_name].quantile([0.25, 0.5, 0.75])
        percentile_25 = percentiles[0.25]
        percentile_50 = percentiles[0.5]
        percentile_75 = percentiles[0.75]
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")

    df['Category'] = np.where(df[column_name] <= percentiles[0.25], 'D',
                              np.where(df[column_name] <= percentiles[0.5], 'C',
                                       np.where(df[column_name] <= percentiles[0.75], 'B', 'A')))

    # Création de la carte choroplèthe
    fig = px.choropleth(df, locations='Country Name', locationmode='country names',
                        color='Category', hover_data=['Country Name', 'Category'],
                        title='Country Categories')

    # Affichage de la carte choroplèthe
    fig.show()

    # Sauvegarde du DataFrame dans un fichier CSV
    repertoire_destination = "statistiquePerPays"
    chemin_repertoire = os.path.join(os.getcwd(), repertoire_destination)

    if not os.path.exists(chemin_repertoire):
        os.makedirs(chemin_repertoire)

    chemin_fichier = os.path.join(chemin_repertoire, nom_fichier)
    df.to_csv(chemin_fichier, float_format='%.2f', index=True)

    #print("Le fichier a été sauvegardé avec succès dans le répertoire statistiqueParPays.")
