import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import altair as alt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit import session_state as _state

_PERSIST_STATE_KEY = f"{__name__}_PERSIST"


def persist(key: str) -> str:
    """Mark widget state as persistent."""
    if _PERSIST_STATE_KEY not in _state:
        _state[_PERSIST_STATE_KEY] = set()

    _state[_PERSIST_STATE_KEY].add(key)

    return key


initial_state = {"uploaded": True}

initial_state_data = {"uploaded": False}


def load_widget_state():
    """Load persistent widget state."""
    if _PERSIST_STATE_KEY in _state:
        _state.update(
            {
                key: value
                for key, value in _state.items()
                if key in _state[_PERSIST_STATE_KEY]
            }
        )


def reset_application_state():
    global _state
    # Сохранение значений, которые не должны быть удалены
    preserved_values = {
        key: _state[key] for key in ["data", "original_data"] if key in _state
    }

    # Очистка текущего состояния
    _state.clear()

    # Загрузка исходного состояния
    _state.update(
        initial_state.copy()
    )  # Используем .copy() для избежания изменений в исходном словаре

    # Восстановление сохраненных значений
    _state.update(preserved_values)
    st.rerun()


def reset_application_state_with_data():
    global _state

    _state.update(
        initial_state_data.copy()
    )  # Используем .copy() для избежания изменений в исходном словаре

    _state.clear()

    st.rerun()


def upload_file():
    """
    Presents a file uploader widget and returns the uploaded file object.

    This function creates a file uploader in a Streamlit app with the specified prompt message.
    Users can upload a file through the UI. If a file is uploaded, the function returns the
    file object provided by Streamlit, allowing further processing of the file. If no file is
    uploaded, the function returns None.

    Returns:
    - uploaded_file (UploadedFile or None): The file object uploaded by the user through the
      file uploader widget. The object contains methods and attributes to access and read the file's
      content. Returns None if no file has been uploaded.

    Example usage:
    ```
    uploaded_file = upload_file()
    if uploaded_file is not None:
        # Process the file
        st.write("File uploaded:", uploaded_file.name)
    else:
        st.write("No file uploaded.")
    ```
    """
    uploaded_file = st.file_uploader("Загрузите файл")
    if uploaded_file is not None:
        return uploaded_file
    return None


def check_nulls(df):
    """
    Checks and summarizes the number and percentage of missing (null) values in each column of a pandas DataFrame.

    Parameters:
    - data (pd.DataFrame): The DataFrame to be analyzed for null values.

    Returns:
    - pd.DataFrame: A DataFrame with two columns: 'amount' which represents the count of null values per column,
      and 'percentage' which shows the percentage of null values per column, rounded to four decimal places and
      multiplied by 100 to convert to percentage format.
    """

    df_null = pd.DataFrame(
        {
            "Missing Values": df.isna().sum().values,
            "% of Total Values": 100 * df.isnull().sum() / len(df),
        },
        index=df.isna().sum().index,
    )
    df_null = df_null.sort_values("% of Total Values", ascending=False).round(2)
    # print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"
    # "There are " + str(df_null.shape[0]) + " columns that have missing values.")
    return df_null


def delete_columns(data, columns):
    """
    Removes specified columns from a DataFrame.

    This function takes a DataFrame and a list of column names to be removed. It returns a new DataFrame with the specified
    columns removed, leaving the original DataFrame unchanged.

    Parameters:
    - data (pd.DataFrame): The original DataFrame from which columns will be removed.
    - columns (list of str): A list of strings representing the names of the columns to be removed.

    Returns:
    - pd.DataFrame: A new DataFrame with the specified columns removed.
    """
    dataframe = data.drop(columns, axis=1)
    return dataframe


def fill_numerical_data(data, columns):
    """
    Fills missing values in specified numerical columns with the median of each column.

    This function iterates over a list of columns in a DataFrame, filling in missing (NaN) values in those columns with
    the median of each respective column. The changes are made in place, but the DataFrame is also returned for convenience
    and chaining operations.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the columns to be filled.
    - columns (list of str): A list of column names (strings) in the DataFrame that are to be filled. These should be columns with numerical data.

    Returns:
    - pd.DataFrame: The original DataFrame with missing values in the specified columns filled with their respective medians.
    """
    for col in columns:
        data[col].fillna(data[col].median(), inplace=True)
    return data


def fill_categorical_data(data, columns):
    """
    Fills missing values in specified categorical columns with the most frequent value of each column.

    Utilizes the SimpleImputer class from sklearn.impute to fill missing (NaN) values in the DataFrame's categorical columns
    with the most frequent value (mode) found in each column. This operation updates the DataFrame in place and returns it
    for potential chaining of operations or further modifications.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the columns to be filled. It's expected to be modified in place.
    - columns (list of str): A list of column names (strings) that are to be treated, which should correspond to categorical data.

    Returns:
    - pd.DataFrame: The original DataFrame with missing values in the specified columns filled with their most frequent value.

    Note:
    - This function requires the sklearn library for the SimpleImputer class. Ensure sklearn is installed and imported correctly.
    """
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="most_frequent")
    for col in columns:
        data[col] = imputer.fit_transform(data[[col]]).ravel()
    return data


def create_toggle(session_name: str, name: str, status=False) -> bool:
    """
    Creates a toggle switch in a Streamlit app and manages its state across reruns.

    This function displays a toggle switch with the label specified by the 'name' parameter.
    It uses Streamlit's session state to keep track of the toggle's position (on/off) across reruns.
    The current state of the toggle is stored in `st.session_state` using a key provided by the
    'session_name' parameter.

    Parameters:
    - session_name (str): The key used to store the toggle's state in `st.session_state`.
                          This key should be unique to each toggle to prevent state conflicts.
    - name (str): The label displayed next to the toggle switch in the UI.

    Returns:
    - bool: The current state of the toggle (True for on, False for off).

    Example:
    ```
    delete_column = create_toggle('delete_status', 'Удалить')
    if delete_column:
        st.write("Toggle is ON - deletion logic can be placed here.")
    else:
        st.write("Toggle is OFF - no deletion occurs.")
    ```
    """
    # Initialize toggle status in session_state if it doesn't exist
    if session_name not in st.session_state:
        st.session_state[session_name] = False

    # Display the toggle and assign its current value based on session_state
    toggle = st.toggle(name, value=st.session_state[session_name], disabled=status)

    # Update session state based on the toggle's position
    if toggle != st.session_state[session_name]:
        st.session_state[session_name] = toggle
        st.rerun()
    return toggle


def apply_one_hot_encoder(df, columns):
    dataframe = pd.get_dummies(df, columns=columns, drop_first=True)
    return dataframe


def apply_label_encoder(df, columns):
    label_encoder = LabelEncoder()
    for column in columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df


def apply_ordinal_encoder(df, columns):
    ordinal_encoder = OrdinalEncoder()
    for column in columns:
        # Ensure the column is in a 2D array format and fit_transform
        # Also, handling the case where column data might not be numeric
        df[column] = ordinal_encoder.fit_transform(df[[column]])
    return df


def reset_all_parameters_selection():
    st.session_state["all_parameters_selected"] = False


def apply_min_max_scaling(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def apply_standardization(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def apply_log_transformation(df, columns):
    # Apply log transformation with a small constant to avoid log(0)
    for column in columns:
        df[column] = np.log(df[column] + 1)
    return df


# def line_chart(data, x_axis, y_axis, plot_title=None, hue=None):
#     fig = px.line(data, x=x_axis, y=y_axis, title=plot_title, color=hue)
#     st.plotly_chart(fig, use_container_width=True)

# def scatter_plot(data, x_axis, y_axis, plot_title=None, hue=None):
#     fig = px.scatter(data, x=x_axis, y=y_axis, title=plot_title, color=hue)
#     st.plotly_chart(fig, use_container_width=True)


def line_chart(data, x_axis, y_axis, input_color_theme, hue=None):
    # If hue is provided, color the circles based on the hue column and the input color theme.
    # If hue is None, all circles will be colored red.
    if hue:
        color_encoding = alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
    else:
        color_encoding = alt.value(
            input_color_theme
        )  # Directly set the color to red if hue is not provided

    c = (
        alt.Chart(data)
        .mark_line()
        .encode(x=x_axis, y=y_axis, color=color_encoding)
        .interactive()
    )

    st.altair_chart(c, use_container_width=True)


def scatter_plot(data, x_axis, y_axis, input_color_theme, hue=None):
    # If hue is provided, color the circles based on the hue column and the input color theme.
    # If hue is None, all circles will be colored red.
    if hue:
        color_encoding = alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
    else:
        color_encoding = alt.value(
            input_color_theme
        )  # Directly set the color to red if hue is not provided

    c = (
        alt.Chart(data)
        .mark_circle()
        .encode(x=x_axis, y=y_axis, color=color_encoding)
        .interactive()
    )

    st.altair_chart(c, use_container_width=True)


import altair as alt
import streamlit as st


def histogram(data, x_axis, input_color_theme, hue=None):
    # If hue is provided, color the bars based on the hue column and the input color theme.
    # If hue is None, all bars will be colored with the input color theme.
    if hue:
        color_encoding = alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
    else:
        color_encoding = alt.value(
            input_color_theme
        )  # Directly set the color to red if hue is not provided

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(x=alt.X(x_axis, bin=True), y="count()", color=color_encoding)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def box_plot(data, x_axis, y_axis, input_color_theme, hue=None):
    # If hue is provided, color the box plot based on the hue column and the input color theme.
    # If hue is None, all plots will be colored with the input color theme.
    if hue:
        color_encoding = alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
    else:
        color_encoding = alt.value(
            input_color_theme
        )  # Directly set the color to red if hue is not provided

    chart = (
        alt.Chart(data)
        .mark_boxplot()
        .encode(x=x_axis, y=y_axis, color=color_encoding)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def density_plot(data, x_axis, input_color_theme, hue=None):
    # Start with the base chart
    chart = alt.Chart(data)

    # Apply the density transformation conditionally based on hue
    if hue:
        # If hue is provided, calculate density with grouping
        chart = chart.transform_density(
            density=x_axis,
            as_=[x_axis, "density"],
            groupby=[hue],  # Ensure this is a list
        )
    else:
        # If hue is not provided, calculate density without grouping
        chart = chart.transform_density(density=x_axis, as_=[x_axis, "density"])

    # Apply the rest of the encoding
    chart = chart.mark_area().encode(
        x=f"{x_axis}:Q",
        y="density:Q",
        color=(
            alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
            if hue
            else alt.value(input_color_theme)
        ),
    )

    st.altair_chart(chart, use_container_width=True)


def custom_linear_regression():
    model = LinearRegression()
    return model


def custom_logistic_regression(C_c=1, n_jobs_c=-1, max_iter_c=100, l1_ratio_c=None):
    model = LogisticRegression(
        C=C_c, n_jobs=n_jobs_c, max_iter=max_iter_c, l1_ratio=l1_ratio_c
    )
    return model


def custom_decision_tree_regression(
    criterion_c="squared_error",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = DecisionTreeRegressor(
        criterion=criterion_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model


def custom_decision_tree_classification(
    criterion_c="gini",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = DecisionTreeClassifier(
        criterion=criterion_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model


def custom_train_test_split(X, y, size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
    return X_train, X_test, y_train, y_test


def confusion_matrix_visualization(y_true, y_pred, input_color="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=input_color, cbar=False)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    st.pyplot(fig)


def regression_result_visualization(y_true, y_pred, input_color):
    # Create a lmplot with seaborn. 'palette' should be a dictionary mapping levels of the hue variable to matplotlib colors.
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, c=input_color)

    st.pyplot(fig)


def correlation_matrix_visualize(data, color):
    fig = plt.figure(figsize=(10, 8))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap=color)
    plt.title("Correlation of Features")

    st.pyplot(fig)


def train_and_predict(model, X_train, y_train, X_test):
    """
    Trains a model and predicts outcomes for the given test data.
    This function uses Streamlit's memoization to cache the model training and prediction steps,
    improving performance for repeated calls with unchanged data.

    Args:
        model: The machine learning model to be trained. Must be hashable by Streamlit.
        X_train (pd.DataFrame or np.ndarray): Training data features.
        y_train (pd.Series or np.ndarray): Training data labels/targets.
        X_test (pd.DataFrame or np.ndarray): Test data features.

    Returns:
        A tuple containing:
        - model: The trained machine learning model.
        - pred: Predictions made by the model on X_test.
    """
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, pred


def custom_random_forest_regression(
    n_estimators_c=100,
    criterion_c="squared_error",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = RandomForestRegressor(
        n_estimators=n_estimators_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model


def custom_random_forest_classification(
    n_estimators_c=100,
    criterion_c="squared_error",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = RandomForestClassifier(
        n_estimators=n_estimators_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model


def custom_svr(kernel_c="rbf", degree_c=3):
    model = SVR(kernel=kernel_c, degree=degree_c)
    return model


def custom_svc(kernel_c="rbf", degree_c=3):
    model = SVC(kernel=kernel_c, degree=degree_c)
    return model


def custom_gbc(
    loss_c="log_loss",
    learning_rate_c=0.1,
    n_estimators_c=100,
    criterion_c="squared_error",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = GradientBoostingClassifier(
        loss=loss_c,
        learning_rate=learning_rate_c,
        n_estimators=n_estimators_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model


def custom_gbr(
    loss_c="squared_error",
    learning_rate_c=0.1,
    n_estimators_c=100,
    criterion_c="squared_error",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = GradientBoostingRegressor(
        loss=loss_c,
        learning_rate=learning_rate_c,
        n_estimators=n_estimators_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model
