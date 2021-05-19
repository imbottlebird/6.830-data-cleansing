import streamlit as st
import awesome_streamlit as ast
import hashlib
import sqlite3
import pandas as pd
import numpy as np
import datetime
import defSessionState as ss
import random
import sklearn

from utils import *

st.set_page_config(
    # Can be "centered" or "wide". In the future also "dashboard", etc.
    layout="centered",
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    # String or None. Strings get appended with "• Streamlit".
    page_title="MIT 6.830 Final Project",
    page_icon=None,  # String, anything supported by st.image, or None.
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# ===========================================
# Data load functions
# ===========================================

import os
import base64

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def main():
    """Main function of the App"""
    st.title('MIT 6.830 - Auto-cleaning Database')
    st.markdown(
        "**By ByeongJo Kong, Kristin YiJie Chen, Ming Da Li**")
    st.markdown(
        "Spring 2021")

    st.info(
        """
        \nData Imputation Process:
            \n1. Upload or Select Data
            \n2. Data Exploration (Scan or Generate Missing Values)
            \n3. Model Performance Test (with a small portion of data)
            \n4. Data Imputation and Evaluation
            \n5. Data Download
        """
    )
    state = ss._get_state()

    ###### DATA UPLOAD #######
    st.title("1. Data Upload/Selection")
    uploaded_file = st.file_uploader(
        "Upload your CSV file here",
        type="csv",
        key='uploaded_file')

    df = pd.DataFrame
    leng = 0
    option = st.selectbox(
    'Or select an example data:',
    ('Please select the data','Abalone', 'Diamonds'))
    if option == 'Abalone':
        df = pd.read_csv('data/abalone.csv')
        st.text("Abalone Data loaded successfully!")
        st.dataframe(df)
    elif option == 'Diamonds':
        df = pd.read_csv('data/diamonds.csv')
        st.text("Diamonds Data loaded successfully!")
        st.dataframe(df)
    # elif option == 'Iris':
    #     df = pd.read_csv('data/iris_data.csv')
    #     st.text("Iris Data loaded successfully!")
    #     st.dataframe(df)
    
    if uploaded_file:
        data_load_state = st.text('Loading Data...')
        df = pd.read_csv(uploaded_file)
        #df = getDataFromCSV(uploaded_file).copy()
        data_load_state.text("Data loaded successfully!")
        st.dataframe(df)

    ###### CHECK MISSING DATA #######
    st.title("2. Data Exploration")
    #check for missing values
    checkMissingData = st.checkbox(
                label='Check for missing values',
                value=False,
                key='checkMissingData')

    if (checkMissingData):
        dfInfo = pd.DataFrame()
        dfInfo["Types"] = df.dtypes
        dfInfo["Missing Values"] = df.isnull().sum()
        dfInfo["Missing Values % "] = (df.isnull().sum()/len(df)*100)
        st.table(dfInfo)


    #adding missing values
    addMissingData = st.checkbox(
                label='Generate missing values in 2 columns',
                value=False,
                key='addMissingData')
    
    #missing_is = random.sample(range(len(df.iloc[0])),2)
    missing_is = [1,4]
    if (addMissingData):
        #missing_is = random.sample(range(len(df.iloc[1])), 2)
        df = add_missing_values(df, missing_is, 0.2)
        df_mv = df.copy()
        st.text("Missing data generated successfully!")
        
        st.dataframe(df)
        dfInfo = pd.DataFrame()
        dfInfo["Types"] = df.dtypes
        dfInfo["Missing Values"] = df.isnull().sum()
        dfInfo["Missing Values % "] = (df.isnull().sum()/len(df)*100)
        st.table(dfInfo)

    ###### PERFORMANCE TEST ######
    if not df.empty:
        st.title("3. Model Performance Test")

        st.write("Preliminary Analysis on the Imputation Model Performance. Select the percentage of data you want to test the model with.")
        input_perc = st.number_input('Percentage of data',0.0,1.0,0.2)
        #st.write('The current number is ', debt_level)
        
        #check for missing values
        TestModel = st.checkbox(
                    label='Test Model Performance',
                    value=False,
                    key='TestModel')

        if (TestModel):
            perc = int(len(df)*input_perc)
            df_impute, acc, score_dict, time_dict = impute_missing_values(df[:perc], missing_is)

            st.markdown("**The best order of imputation:**")
            st.write('Column ', acc['best imputation order'][0]+1, ', Column ',acc['best imputation order'][1]+1)

            st.markdown("**Types of task conducted:**")
            if (type(acc['best imputation model'][0]) == sklearn.linear_model._base.LinearRegression or type(acc['best imputation model'][0]) == sklearn.ensemble._forest.RandomForestRegressor):
                st.write('Column ', acc['best imputation order'][0]+1, ': Regression task')
            else:
                st.write('Column ', acc['best imputation order'][0]+1, ': Classification task')

            if (type(acc['best imputation model'][1]) == sklearn.linear_model._base.LinearRegression or type(acc['best imputation model'][1]) == sklearn.ensemble._forest.RandomForestRegressor):
                st.write('Column ', acc['best imputation order'][1]+1, ': Regression task')
            else:
                st.write('Column ', acc['best imputation order'][1]+1, ': Classification task')
                
            # st.write('Column ', acc['best imputation order'][0], ': ',acc['best imputation model'][0])
            # st.write('Column ', acc['best imputation order'][1], ': ',acc['best imputation model'][1])

            # if score_dict['Logistic'] != 0:
            #     st.markdown('This is a **Classification Task**')
            # else:
            #     st.markdown('This is a **Regression Task**')
            ds = [score_dict, time_dict]
            d = {}
            for k in score_dict.keys():
                d[k] = tuple(d[k] for d in ds)
            st.table(pd.DataFrame(d,index=['Mean Accuracy','Time']))
            st.dataframe(df_impute)

        ###### DATA IMPUTATION ######
        st.title("4. Data Imputation and Evaluation")
        mod_option = st.selectbox(
        'Select your imputation model: (Only Automatic option is available at the moment.)',
        ('Please select the model','Linear Regression', 'Logistic Regression', 'Random Forest', 'Automatic'))
        if mod_option == 'Linear Regression':
            # df = pd.read_csv('data/abalone.csv')
            # st.text("Abalone Data loaded successfully!")
            # st.dataframe(df)
            st.text("Currently only Automatic option is available.")
        elif mod_option == 'Logistic Regression':
            # df = pd.read_csv('data/diamonds.csv')
            # st.text("Diamonds Data loaded successfully!")
            # st.dataframe(df)
            st.text("Currently only Automatic option is available.")
        elif mod_option == 'Random Forest':
            # df = pd.read_csv('data/iris_data.csv')
            # st.text("Iris Data loaded successfully!")
            # st.dataframe(df)
            st.text("Currently only Automatic option is available.")
        elif mod_option == 'Automatic':
            data_load_state2 = st.text('Imputing the missing values..')
            # st.text("Imputing the missing values..")
            df_impute, acc, score_dict, time_dict = impute_missing_values(df, missing_is)
            # st.write(max(score_dict, key=score_dict.get), 'was selected to impute the missing data!')
            
            st.markdown("**The best order of imputation:**")
            st.write('Column ', acc['best imputation order'][0]+1, ', Column ',acc['best imputation order'][1]+1)

            st.markdown("**Model Recommendation**")
            if (type(acc['best imputation model'][0]) == sklearn.linear_model._base.LinearRegression or type(acc['best imputation model'][0]) == sklearn.ensemble._forest.RandomForestRegressor):
                st.write('Column ', acc['best imputation order'][0]+1, '(Regression task) :', acc['best imputation model'][0])
            else:
                st.write('Column ', acc['best imputation order'][0]+1, '(Classification task) :', acc['best imputation model'][0])

            if (type(acc['best imputation model'][1]) == sklearn.linear_model._base.LinearRegression or type(acc['best imputation model'][1]) == sklearn.ensemble._forest.RandomForestRegressor):
                st.write('Column ', acc['best imputation order'][1]+1, '(Regression task) :', acc['best imputation model'][1])
            else:
                st.write('Column ', acc['best imputation order'][1]+1, '(Classification task) :', acc['best imputation model'][1])
            
            ds = [score_dict, time_dict]
            d = {}
            for k in score_dict.keys():
                d[k] = tuple(d[k] for d in ds)
            data_load_state2.text("Data loaded successfully!")
            # st.text("Automatic imputation completed!")
            st.table(pd.DataFrame(d,index=['Mean Accuracy','Time']))
            st.text("Missing values are imputed based on the recommended setting!")
            st.dataframe(df_impute)

        #### FILE DOWNLOAD ####
        st.title("5. Data Download")
        if st.button('Download the imputed data as CSV'):
            tmp_download_link = download_link(df, 'imputed_data.csv', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)



if __name__ == "__main__":
    main()
