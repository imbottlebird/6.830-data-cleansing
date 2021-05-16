import streamlit as st
import awesome_streamlit as ast
import hashlib
import sqlite3
import pandas as pd
import numpy as np
import datetime
import defSessionState as ss

st.set_page_config(
    # Can be "centered" or "wide". In the future also "dashboard", etc.
    layout="centered",
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    # String or None. Strings get appended with "• Streamlit".
    page_title="MIT 6.830 Final Project",
    page_icon=None,  # String, anything supported by st.image, or None.
)


# ===========================================
# Data load functions
# ===========================================

@st.cache(allow_output_mutation=True)
def getDataFromCSV(file) -> pd.DataFrame:
    dataFrame = pd.read_csv(file, sep=",", decimal=".",
                     encoding="UTF-8", 
                     index_col=0,
                     low_memory=False)
    dataFrame.index = np.arange(1, len(dataFrame) + 1)#pd.to_datetime(dataFrame.index, format='%Y-%m-%d %H:%M:%S')
    #dataFrame = dataFrame.sort_index(ascending=True)
    dataFrame = dataFrame.apply(pd.to_numeric, errors='coerce')
    return dataFrame

def main():
    """Main function of the App"""
    st.title('MIT 6.830 Project - Auto-cleaning Database')
    st.markdown(
        "Upload the CSV file or Select a sample dataset from the list below.")

    st.info(
        """
        \nList of testing data sets for machine learning imputation methods:
            \n* Abalone
            \n* Diamonds
            \n* IOT_tempt
            \n* Iris
            \n* Msleep
            \n* Mtcars
            \n* Travel_time
        """
    )
    state = ss._get_state()

    uploaded_file = st.file_uploader(
        "Upload your CSV file here",
        type="csv",
        key='uploaded_file')

    if uploaded_file:
        data_load_state = st.text('Loading Data...')
        df = pd.read_csv(uploaded_file)
        #df = getDataFromCSV(uploaded_file).copy()
        data_load_state.text("Data loaded successfully!")
        st.dataframe(df)


        if not df.empty:
            st.title("Data Preparation")
            
            expanderFltTags = st.beta_expander(
                label='Select your variables to analyze',
                expanded=False)

            state.dfTags = df.columns.values.tolist()

            state.fltTags = expanderFltTags.multiselect(
                label='Variables to process:',
                options=state.dfTags,
                default=state.fltTags,
                key='fltTagsPreparation')

            dfRaw = df[state.fltTags]

            startTimeDf = df.index[0]
            endTimeDf = df.index[-1]

            selStart = startTimeDf
            selEnd = endTimeDf

            dfRawRange = dfRaw.loc[selStart:selEnd]
            state.dfRawRange = dfRawRange

            if state.fltTags != []:

                st.markdown("------------------------------------------")
                st.markdown("Select the options below to process the data.")
                
                #################
                ### Dataframe ###
                #################

                showRawData = st.checkbox(
                    label='Data View',
                    value=False,
                    key='showRawData')

                if (showRawData):
                    st.dataframe(data=dfRawRange)

                ############
                ### Info ###
                ############

                showInfo = st.checkbox(
                    label='Dataframe Information with Missing Values', value=False, key='showInfo')
                if (showInfo):

                    dfInfo = pd.DataFrame()
                    dfInfo["Types"] = dfRawRange.dtypes
                    dfInfo["Missing Values"] = dfRawRange.isnull().sum()
                    dfInfo["Missing Values % "] = (dfRawRange.isnull().sum()/len(dfRawRange)*100)
                    st.table(dfInfo)

                ################
                ### Cleaning ###
                ################

                state.execCleaning = st.checkbox(label='Automatic Data Cleaning', value= (state.execCleaning or False), key='execCleaning')
                if (state.execCleaning):

                    methodCleaning = ["Data Imputation", "Drop NaN"]
                    selectCleaning = st.selectbox(label='Select the Cleaning Method',
                                                  options=methodCleaning,
                                                  key='selectCleaning')
                    if selectCleaning == "Drop NaN":
                        dfCleaned = dfRawRange.dropna(how="any")
                        
                    elif selectCleaning == "Data Imputation":
                        methodImputation = ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]
                        selectImputation = st.selectbox(label='Select the Imputation Method',
                                                           options=methodImputation,
                                                           key='selectImputation')
                        dfCleaned = dfRawRange.interpolate(method=selectImputation, inplace=False)
                    st.text("Dataframe Information after Cleaning.")
                    
                    state.dfRawRange = dfCleaned
                    
                    dfInfo = pd.DataFrame()
                    dfInfo["Types"] = state.dfRawRange.dtypes
                    dfInfo["Missing Values"] = state.dfRawRange.isnull().sum()
                    dfInfo["Missing Values % "] = (state.dfRawRange.isnull().sum()/len(state.dfRawRange)*100)

                    st.table(dfInfo)

                ################
                ### Describe ###
                ################

                showDescribe = st.checkbox(
                    label='Show Descriptive Statistics', value=False, key='showDescribe')
                if (showDescribe):
                    st.table(dfRawRange.describe().transpose())

                ##########################
                ### Model Evaluation ###
                ##########################

                showInfoVar = st.checkbox(
                    label="Evaluation by Model",\
                        value=False, 
                        key='showInfoVar')
                if (showInfoVar):
                    
                    st.write("Select variables")
                    fltPlot = st.selectbox(label='', options=state.fltTags, key='fltPlot')

    

if __name__ == "__main__":
    main()
