# Load libraries
import psycopg2 as pg
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import pickle


# Set random seed
np.random.seed(0)

connection = pg.connect(
    host='localhost',
    port=54320,
    dbname='trans_db',
    user='postgres'
)

# calculate mean and standard deviation


def calculate_mean_sd():
    tb_1 = 'transactions'
    tb_2 = 'fraudsters'
    ddl = f""" SELECT tb_1.USER_ID, SUM(AMOUNT_USD) as Amount_Sum, AVG(AMOUNT_USD) as Amount_Mean, stddev_samp(AMOUNT_USD) as Amount_Standard_Deviation FROM {tb_1} AS tb_1 LEFT JOIN {tb_2} AS tb_2 ON tb_1.USER_ID = tb_2.USER_ID Where tb_2.USER_ID is NULL GROUP BY tb_1.USER_ID ORDER BY tb_1.USER_ID"""
    return pd.read_sql(ddl, con=connection)

# calculate median


def calculate_median():
    tb_1 = 'transactions'
    tb_2 = 'fraudsters'
    ddl = f""" SELECT DISTINCT tb_1.USER_ID, td.p_50 AS Amount_Median from {tb_1} as tb_1 join (SELECT tb_1.USER_ID, percentile_cont(0.5) within group (order by tb_1.AMOUNT_USD) as p_50 from {tb_1} as tb_1 LEFT JOIN {tb_2} AS tb_2 ON tb_1.USER_ID = tb_2.USER_ID Where tb_2.USER_ID is NULL group by tb_1.USER_ID ) td on tb_1.USER_ID = td.USER_ID """
    return pd.read_sql(ddl, con=connection)

# fetch data from database for machine learning model


def data_fetch():
    tb_1 = 'transactions'
    tb_2 = 'fraudsters'
    tb_3 = 'currency_details'
    tb_4 = 'countries'
    tb_5 = 'users'

    # transaction data
    ddl = f""" SELECT tb_1.*, tb_2.user_id as Fraudster, tb_3.* FROM {tb_1} AS tb_1 LEFT JOIN {tb_2} AS tb_2 ON tb_1.USER_ID = tb_2.USER_ID LEFT JOIN {tb_3} AS tb_3 ON tb_1.CURRENCY = tb_3.CCY """
    transactions_details = pd.read_sql(ddl, con=connection)

    # countries data
    ddl = f""" SELECT * FROM {tb_4} """
    countries_details = pd.read_sql(ddl, con=connection)

    # user data
    ddl = f""" SELECT * FROM {tb_5}  """
    users_details = pd.read_sql(ddl, con=connection)

    return transactions_details, countries_details, users_details


def cols_upper(df, cols_to_upper):

    for col in cols_to_upper:
        df[col] = df[col].str.upper()

    return df


def feature_extraction(transactions_details, users_details):
    # convert usd in proper number
    transactions_details['amount_usd'] = transactions_details['amount_usd'] * 100

    # feature of dates
    transactions_details['month'] = transactions_details['created_date'].dt.month
    transactions_details['day'] = transactions_details['created_date'].dt.day
    transactions_details['day_of_week'] = transactions_details['created_date'].dt.dayofweek
    transactions_details['is_weekend'] = transactions_details.day_of_week.isin([
                                                                               5, 6]) * 1
    transactions_details['quarter'] = transactions_details['created_date'].dt.quarter

    # age feature using user data
    current_year = datetime.datetime.now().year
    users_details['age'] = current_year - users_details['birth_year']

    # extracting important user feature
    user_features = [
        'id',
        'has_email',
        'phone_country',
        'is_fraudster',
        'name',
        'terms_version',
        'state',
        'age',
        'kyc',
        'failed_sign_in_attempts',
        'created_date']
    users_details = users_details[user_features]
    # rename features
    users_details.rename(columns={'name': 'user_country'}, inplace=True)
    users_details.rename(columns={'state': 'user_state'}, inplace=True)

    # extracting important transaction feature
    transaction_features = [
        'currency',
        'state',
        'merchant_category',
        'user_id',
        'type',
        'source',
        'entry_method',
        'amount_usd',
        'fraudster',
        'is_crypto',
        'name',
        'month',
        'day',
        'day_of_week',
        'is_weekend',
        'quarter',
        'created_date']
    transactions_details = transactions_details[transaction_features]
    # rename features
    transactions_details.rename(
        columns={
            'name': 'merchant_country'},
        inplace=True)
    transactions_details.rename(
        columns={
            'state': 'transaction_state'},
        inplace=True)
    transactions_details.rename(
        columns={
            'created_date': 'transaction_date'},
        inplace=True)

    # join transaction and user details data
    transactions_details = pd.merge(
        transactions_details,
        users_details,
        left_on='user_id',
        right_on='id',
        how='left')

    # caculate more feature
    # feature of difference in days from making account and doing first
    # transaction
    transactions_details['diff_in_days'] = (
        transactions_details['transaction_date'] -
        transactions_details['created_date']).dt.days

    # find number of null values and its percent
    print (transactions_details.isnull().sum())
    print (transactions_details.isnull().mean().round(4) * 100)

    # drop unused columns
    # drop as more null values like 'merchant_category', 'terms_version'
    # drop as feature is extracted from them 'transaction_date', 'created_date'
    # drop column which are unused 'id','user_id', 'is_fraudster',
    # 'user_state' for fraud detection
    transactions_details = transactions_details.drop(
        labels=[
            'merchant_category',
            'terms_version',
            'transaction_date',
            'created_date',
            'id',
            'user_id',
            'is_fraudster',
            'user_state'],
        axis=1)

    # drop rows having null values in any column
    transactions_details = transactions_details.dropna(how='any', axis=0)

    return transactions_details


def explore_data(transactions_details, users_details):

    # Seperate transaction data into non-fraud and fraud cases
    # save non-fraud df observations into a separate df
    df_nonfraud = transactions_details[transactions_details.fraudster == 0]
    df_fraud = transactions_details[transactions_details.fraudster == 1]

    # Seperate user data into non-fraud and fraud cases
    df_user_nonfraud = users_details[users_details.is_fraudster == False]
    df_user_fraud = users_details[users_details.is_fraudster]

    categorical_features = [
        'has_email',
        'name',
        'terms_version',
        'state',
        'kyc',
        'failed_sign_in_attempts']
    for col in categorical_features:
        print ((df_user_fraud[col].value_counts() /
                df_user_fraud[col].count()) * 100)
        print ((df_user_nonfraud[col].value_counts()/df_user_nonfraud[col].count())*100)

    categorical_features = [
        'state',
        'name',
        'merchant_category',
        'entry_method',
        'source',
        'currency',
        'is_crypto']
    for col in categorical_features:
        print ((df_fraud[col].value_counts() / df_fraud[col].count()) * 100)
        print ((df_nonfraud[col].value_counts()/df_nonfraud[col].count())*100)

    # Plot of high value transactions($500-$1500)
    bins = np.linspace(500, 1500, 100)
    plt.hist(
        df_nonfraud.amount_usd,
        bins,
        alpha=1,
        normed=True,
        label='Non-Fraud')
    plt.hist(df_fraud.amount_usd, bins, alpha=1, normed=True, label='Fraud')
    plt.legend(loc='upper right')
    plt.title(r"Amount by percentage of transactions (transactions \$500-$1500)")
    plt.xlabel("Transaction amount (USD)")
    plt.ylabel("Percentage of transactions (%)")
    #plt.show()


def pre_process(transactions_details, countries_details, users_details):

    # mapping transaction data with 0 and 1 to show fraud transaction or not
    transactions_details['fraudster'] = transactions_details.fraudster.map(
        lambda x: 0 if x is None else 1)

    # columns need to change to upper case
    cols_to_upper = [
        'currency',
        'state',
        'merchant_category',
        'merchant_country',
        'entry_method',
        'type',
        'source']
    transactions_details = cols_upper(transactions_details, cols_to_upper)

    # columns need to change to upper case for user data
    cols_to_upper = ['phone_country', 'state', 'country', 'kyc']
    users_details = cols_upper(users_details, cols_to_upper)

    # columns need to change to upper case for country data
    cols_to_upper = ['code', 'name', 'code3']
    countries_details = cols_upper(countries_details, cols_to_upper)

    # append country data in different format so as to join it with user and transaction data
    # extract differnet codes of country
    code_1 = countries_details[['code', 'name']]
    code_2 = countries_details[['code3', 'name']]
    code_3 = countries_details[['numcode', 'name']]
    code_4 = countries_details[['phonecode', 'name']]

    # rename column names
    code_2.rename(columns={'code3': 'code'}, inplace=True)
    code_3.rename(columns={'numcode': 'code'}, inplace=True)
    code_4.rename(columns={'phonecode': 'code'}, inplace=True)

    # append different code of country
    code_1 = code_1.append(code_2)
    code_1 = code_1.append(code_3)
    countries_details = code_1.append(code_4)

    # join transaction data and country data
    transactions_details = pd.merge(
        transactions_details,
        countries_details,
        left_on='merchant_country',
        right_on='code',
        how='left')

    # join user data and country data
    users_details = pd.merge(
        users_details,
        countries_details,
        left_on='country',
        right_on='code',
        how='left')

    explore_data(transactions_details, users_details)
    transactions_details = feature_extraction(
        transactions_details, users_details)

    return transactions_details

# Define a roc_curve function


def plot_roc_curve(false_positive_rate, true_positive_rate, roc_auc):
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=5,
        label='AUC = %0.3f' %
        roc_auc)
    plt.plot([0, 1], [0, 1], linewidth=5)
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='upper right')
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Define a precision_recall_curve function


def plot_pr_curve(recall, precision, average_precision):
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        '2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()


def train_model(transactions_details):
    X = transactions_details.drop(columns='fraudster')
    y = transactions_details['fraudster'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    categorical_cols = [
        'currency',
        'transaction_state',
        'type',
        'source',
        'entry_method',
        'is_crypto',
        'merchant_country',
        'phone_country',
        'user_country',
        'kyc']
    numerical_cols = [
        'failed_sign_in_attempts',
        'age',
        'diff_in_days',
        'amount_usd']

    # pre processing pipeline
    # Feature Scaling
    # One Hot Encoding
    preprocess = make_column_transformer(
        (make_pipeline(SimpleImputer(), StandardScaler()), numerical_cols),
        (OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    )

    # Create a pipeline
    model = Pipeline([
        ('preprocess', preprocess),
        ('sampling', SMOTE(random_state=42)),
        ('classification', RandomForestClassifier())
    ])

    # fit model
    model.fit(X_train, y_train)

    # Predict target vector
    y_pred = model.predict(X_test)

    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification report:\n', classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    # Create true and false positive rates
    false_positive_rate, true_positive_rate, threshold = roc_curve(
        y_test, y_pred)

    # Calculate Area Under the Receiver Operating Characteristic Curve
    probs = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, probs[:, 1])
    print('ROC AUC Score:', roc_auc)

    # Obtain precision and recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

    # Calculate average precision
    average_precision = average_precision_score(y_test, y_pred)

    # Plot the roc curve
    plot_roc_curve(false_positive_rate, true_positive_rate, roc_auc)

    # Plot recall precision curve
    plot_pr_curve(recall, precision, average_precision)

    return model


def main():

    # calculate mean median and standard deviation
    mean_sd_df = calculate_mean_sd()
    median_df = calculate_median()

    # save result in csv
    df_result = pd.merge(mean_sd_df, median_df, on='user_id')
    df_result.to_csv('result/result.csv', index=False)

    transactions_details, countries_details, users_details = data_fetch()

    transactions_details = pre_process(
        transactions_details,
        countries_details,
        users_details)

    final_model = train_model(transactions_details)

    # Exporting Model
    with open('result/fraud_detection.pkl', 'wb') as fid:
        pickle.dump(final_model, fid)


if __name__ == "__main__":
    main()
else:
    print ("Executed when imported")
