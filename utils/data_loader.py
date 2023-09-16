import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from utils.preprocessing import message_encodding_and_timecolor


def reddit_loader(input_files_path, input_file, text_column, modified_text_column, prefix_column_name, sentemb_name, \
                  number_of_colors, utc_columnname, time_color_columnname, start_day, end_day, save_delim, \
                  start_feature_column=593, end_feature_column=1361, color_column=1361):
    '''
    Loading Reddit dataset

    Args:
        input_files_path (str): folder path for the input file
        input_file (str): name of the input file
        text_column (str): text column name
        modified_text_column (str): cleaned text text column name
        prefix_column_name (str): prefix name (that will followed by "_number" for each embedding column)
        sentemb_name (str): sentence embedding type / name
        number_of_colors (int): number of colors
        utc_columnname (str): utc data type column name (creation time)
        time_color_columnname (str): new timecolor column name to be created
        start_day (str): start day in a format YYYY-MM-DD
        end_day (str): end day in a format YYYY-MM-DD
        save_delim (str): delimiter separation (default ',')
        start_feature_column (int): column id of the feature start
        end_feature_column (int): column id of the feature end
        color_column (int): color column
    Returns:
        data (np.array): numpy array of encoded messages
        colors_data (no.array): numpy 1D array of colors of the messages
        euclid_dist (np.array): square numpy array of euclidean distances between the messages


    '''
    df_input = message_encodding_and_timecolor(input_files_path, input_file, text_column, modified_text_column, \
                                        prefix_column_name, sentemb_name, number_of_colors, utc_columnname, \
                                        time_color_columnname, start_day, end_day, save_delim)


    # shuffle the dataset before running
    df_input = df_input.sample(frac=1, random_state=42).reset_index(drop=True)
    list_columns = list(df_input.columns)
    print("List columns:", [(i, list_columns[i]) for i in range(len(list_columns))])
    feature_slices = slice(start_feature_column, end_feature_column)
    features_list = list_columns[feature_slices]
    print("Encoding columns", features_list)
    df_feat = df_input[features_list]
    df_colors = df_input[list_columns[color_column]]
    df_num = df_feat.iloc[:, :]
    data = df_num.to_numpy()
    colors_data = df_colors.to_numpy()
    print("Data shape", data.shape, colors_data.shape)
    euclid_dist = euclidean_distances(data)
    print("Distances matrix shape:", euclid_dist.shape)

    return data, colors_data, euclid_dist


def movielens_loader(input_files_path, input_file, text_column, modified_text_column, prefix_column_name, sentemb_name, \
                   number_of_colors, utc_columnname, time_color_columnname, start_day, end_day, save_delim, \
                   start_feature_column=578, end_feature_column=1345, color_column=1345):
    '''
    Loading MoveLens dataset

    Args:
        input_files_path (str): folder path for the input file
        input_file (str): name of the input file
        text_column (str): text column name
        modified_text_column (str): cleaned text text column name
        prefix_column_name (str): prefix name (that will followed by "_number" for each embedding column)
        sentemb_name (str): sentence embedding type / name
        number_of_colors (int): number of colors
        utc_columnname (str): utc data type column name (creation time)
        time_color_columnname (str): new timecolor column name to be created
        start_day (str): start day in a format YYYY-MM-DD
        end_day (str): end day in a format YYYY-MM-DD
        save_delim (str): delimiter separation (default ',')
        start_feature_column (int): column id of the feature start
        end_feature_column (int): column id of the feature end
        color_column (int): color column
    Returns:
        data (np.array): numpy array of encoded messages
        colors_data (no.array): numpy 1D array of colors of the messages
        euclid_dist (np.array): square numpy array of euclidean distances between the messages


    '''
    df_input = message_encodding_and_timecolor(input_files_path, input_file, text_column, modified_text_column, \
                                        prefix_column_name, sentemb_name, number_of_colors, utc_columnname, \
                                        time_color_columnname, start_day, end_day, save_delim)

    df_input = df_input[
        (pd.to_datetime(df_input['date_created']) >= pd.to_datetime(start_day).tz_localize('Etc/GMT')) & (
                    pd.to_datetime(df_input['date_created']) < pd.to_datetime(end_day).tz_localize('Etc/GMT'))]

    df_input = df_input.sample(frac=1, random_state=42).reset_index(drop=True)
    df_input['genre'] = df_input['movie_genres'].str.split(pat="|").apply(lambda x: x[0])
    df_input['genre_color'] = pd.factorize(df_input['genre'])[0]
    df_input['genre_color'] = df_input['genre_color'].astype(int)
    ser_genres = df_input['movie_genres'].str.split(pat="|")
    ser_genres = ser_genres.apply(lambda x: x[0])
    all_generes = set(list(ser_genres))
    print("List genres, ", all_generes, len(all_generes))

    # shuffle the dataset before running
    df_input = df_input.sample(frac=1, random_state=42).reset_index(drop=True)
    list_columns = list(df_input.columns)
    print("List columns:", [(i, list_columns[i]) for i in range(len(list_columns))])
    feature_slices = slice(start_feature_column, end_feature_column)
    features_list = list_columns[feature_slices]
    print("Encoding columns", features_list)
    df_feat = df_input[features_list]
    df_colors = df_input[list_columns[color_column]]
    df_num = df_feat.iloc[:, :]
    data = df_num.to_numpy()
    colors_data = df_colors.to_numpy()
    print("Data shape", data.shape, colors_data.shape)
    euclid_dist = euclidean_distances(data)
    print("Distances matrix shape:", euclid_dist.shape)

    return data, colors_data, euclid_dist
