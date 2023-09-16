import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

nltk.download('punkt')


def index_range_parser(range_string):
    """Parse index range

    Args:
        range_string (string): index ranges seperated by comma like "1-2,3"

    Returns:
        expanded_range (list): expanded indices like [1,2,3]
    """
    ranges = [index.strip() for index in range_string.split(',')]
    expanded_range = []

    for index_range in ranges:
        indices = index_range.split('-')
        start_ind = int(indices[0])
        if len(indices) < 2:
            end_ind = start_ind
        else:
            end_ind = int(indices[1])
        expanded_range.extend(range(start_ind, end_ind + 1))
    return expanded_range


def readinput_file(input_files_path, input_file, save_delim=','):
    '''
    Reading input file

    Args:
        input_files_path (str): folder path for the input file
        input_file (str): name of the input file
        save_delim (str): delimiter separation (default ',')
    Returns:
        df_input (pd.DataFrame): input pandas dataframe
    '''
    df_input = pd.read_csv(os.path.join(input_files_path, input_file), sep=save_delim)
    print("Initial columns", df_input.columns)
    print("Intial shape", df_input.shape)
    print("Reading initial file complete!")
    return df_input


def cleaning_nan(df_input, id_column=None):
    '''
    Cleaning NaN from a dataframe


    Args:
        df_input (pd.DataFrame): input pandas dataframe
        id_column (str): id_column name

    Returns:
        df_input (pd.DataFrame): augmented pandas dataframe with cleaned NaN's
    '''
    print("Shape before cleaning NaN: ", df_input.shape[0])
    if id_column is not None:
        df_input = df_input[~df_input[id_column].isnull()]
    df_input = df_input.reset_index()
    print("Shape after cleaning NaN: ", df_input.shape[0])
    return df_input


def add_time_as_color(df_input, number_of_colors, utc_columnname, time_color_columnname, start_day, end_day):
    '''
    Adding column with the time as color (color assigned on a message based on the creation time)


    Args:
        df_input (pd.DataFrame): input pandas dataframe
        number_of_colors (int): number of colors
        utc_columnname (str): utc data type column name (creation time)
        time_color_columnname (str): new timecolor column name to be created
        start_day (str): start day in a format YYYY-MM-DD
        end_day (str): end day in a format YYYY-MM-DD

    Returns:
        df_input (pd.DataFrame): augmented pandas dataframe containing a column time as color name
    '''

    created_utc_min = pd.to_datetime(start_day).tz_localize('Etc/GMT').timestamp()
    created_utc_max = pd.to_datetime(end_day).tz_localize('Etc/GMT').timestamp()
    coefficient = (created_utc_max - created_utc_min) / number_of_colors

    print("Shape before cleaning utc NULLs", df_input.shape)
    df_input = df_input[~df_input[utc_columnname].isnull()]
    print("Shape after cleaning utc NULLs", df_input.shape)

    df_input[time_color_columnname] = (df_input[utc_columnname].astype(float) - created_utc_min) // coefficient
    df_input[time_color_columnname] = df_input[time_color_columnname].astype(int)
    return df_input


def remove_special_stopwords(df_input, text_column, modified_text_column):
    '''
    Removing special and stop-words

    For a given input dataframe that has a textual column for embedding
    Output a dataframe with the same number of rows and columns augmented with an additional column where the text column
    is cleaned from stopwords

    Args:
        df_input (pd.DataFrame): input pandas dataframe
        text_column (str): text column name
        modified_text_column (str): cleaned text text column name

    Returns:
        df_input (pd.DataFrame): augmented pandas dataframe containing a column with cleaned text
    '''
    stop_words_l = stopwords.words('english')
    df_input[modified_text_column] = df_input[text_column].apply(lambda x: " ".join(
        re.sub(r'[^a-zA-Z]', ' ', w).lower() for w in str(x).split() if
        re.sub(r'[^a-zA-Z]', ' ', w).lower() not in stop_words_l))

    return df_input


def messages_encoding_multidim(df_input, text_column, prefix_column_name="bert_", sentemb_name="bert-base-nli-mean-tokens"):
    '''
    Semantic sentence embedding

    For a given input dataframe that has a textual column for embedding
    Output a dataframe with the same number of rows and columns augmented with the semantic embedding with the new
    embedding where each dimension is a column

    Args:
        df_input (pd.DataFrame): input pandas dataframe containing the column to be semantically embedded
        text_column (str): text column name
        prefix_column_name (str): prefix name (that will followed by "_number" for each embedding column)
        sentemb_name (str): sentence embedding type / name
    Returns:
        df_input_augmented (pd.DataFrame): augmented pandas dataframe containing the embedding (each dimension is s separate column)
    '''
    sentemb_model = SentenceTransformer(sentemb_name)
    message_embeddings = sentemb_model.encode(df_input[text_column])

    columns_sentemb = [prefix_column_name + str(i) for i in range(0, message_embeddings.shape[1])]
    df_sentemb = pd.DataFrame(message_embeddings, columns=columns_sentemb)
    df_input_augmented = pd.concat([df_input, df_sentemb], axis=1)

    return df_input_augmented, message_embeddings


def message_encodding_and_timecolor(input_files_path, input_file, text_column, modified_text_column, \
                                    prefix_column_name, sentemb_name, number_of_colors, utc_columnname, \
                                    time_color_columnname, start_day, end_day, save_delim):
    '''
    Cleaning NaNs, stop words, encode the text into a metric space, and add time as a color

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
    Returns:
        df_input_augmented (pd.DataFrame): augmented pandas dataframe containing the embedding (each dimension is s separate column)
    '''
    df_input = readinput_file(input_files_path, input_file, save_delim=save_delim)
    df_input = cleaning_nan(df_input)
    df_input = remove_special_stopwords(df_input, text_column, modified_text_column)
    df_input, _ = messages_encoding_multidim(df_input, modified_text_column, prefix_column_name=prefix_column_name, \
                                             sentemb_name=sentemb_name)
    df_input_augmented = add_time_as_color(df_input, number_of_colors, utc_columnname, time_color_columnname, \
                                           start_day, end_day)

    return df_input_augmented



