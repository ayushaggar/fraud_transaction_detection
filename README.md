## Objective
1) Fraud transaction detection - Supervised Model
2) Feature Engineering for that

## Data Preparation
1) PostgreSQL 11 server for storing and retrieving data
2) Docker for hosting PostgreSQL

**Input** -
1) Transaction Data
2) Currency Data
3) Country Data
4) User Data with Fraudster label

**Output** :
1) Data is Analysed using SQL
2) Model is saved in pkl format

## Practice
1. Code is production ready from fetching data from SQL till exporting model in pkl format which can be used with any API like Flask
2. Various features are extracted depedning on business use case, given data and its understanding
3. Comments are there in python file and Read me file is provided
4. Techniques implemented can be implement to other business problems as code is done keeping reproducibility of research and analysis done

**Note**: Python code is pep8 compliant


## Tools use 
> Python 3
> Docker 

> Main Libraries Used -
1) numpy
2) pandas
3) scikit
4) imblearn
5) matplotlib
6) yaml
7) psycopg2


## Installing and Running

> 
```sh
$ cd fraud_transaction_detection
$ pip install -r requirements.txt
``` 

For Running Docker Server
1. To run the server:
```sh
$ docker run -d --name data_server -v trans_dbdata:/var/lib/postgresql/data -p 54320:5432 postgres:11
```
2. Create the database:
```sh
$ docker exec -it data_server psql -U postgres -c "create database trans_db"
```

For Running Script
1. For Load data
```sh
$ python etl.py
```
2. For Analyse data
```sh
$ python analyse.py
```

## Various steps in approach are -
1. Data are loaded in PostgreSQL using specified schema
2. Data wrangling using SQL
	1. Fraudsters data is joined with transaction that using user id so as to find fraud transaction
	2. Countries data are transformed so as to join with transaction and user data which have countries in both 4 type of  alpha-numeric representations. It is used to standardise country codes to one format 'name' here
3. Data transformation for supervised model for transaction level feature set (Feature Engineering)
    Features are -
    1. User details - failed_sign_in_attempts, kyc, user country, phone country, has email
    2. Diff_in_days - Difference in days between when user created and transaction date
    3. Age - Age of the user using created date
	4. Transaction Date Features - quarter, is_weekend, day_of_week, day, month
	5. Amount Details - currency used, type, source, entry_method, is_crypto, amount_usd [converted into actual unit using exponent in currency details]
	6. Transaction state - Declined or not
	7. Many features are droped due to less data for all transaction like terms_version, merchant_category
	8. Many features are droped due to insignificant for fraud in transaction like state of user locked or not as if locked there will be no transaction
	9. Many rows are droped to which we do not have user data like has email, user country, age, kyc, failed_sign_in_attempts, diff_in_days
3. Building machine learning model and evaluation. It includes
    1. Scaling (StandardScaler) - It is used to bring all numerical features at the same scale
	2. One Hot Encoding - It is used to bring categorical features to numerical format so as to use ML model
	3. SMOTE - It is used for over sampling as number of fraudster transaction is less in the data
	4. RandomForestClassifier - It is used as an ensemble method and take care of overfitting
    5. Confusion matrix and classification report is calculated so as to find recall for fraudster 

Various Techniques implied for preprocessing -
1. Remove unused columns for decreasing data storage
2. Rename Columns to standardised name for easy use
3. Taken care of edge cases while input for handling error like removing null values in categorical column
4. String change to upper case so as to group unique values
5. Date parameter into datetime format for easy computation
6. Check proportion of null values
7. Remove rows having null values in columns so that analysis can be done taking that features

## Result
1. Part 1 -> SQL ->
	Query to calculate the mean, standard deviation and median transaction amounts
2. Part 2 ->  
	1. DATA EXPLORATION ->
		Goel is to find transacation which can be fraud depending on user details and transaction details
		Analyse number of fraudster transaction for each type of category column to have understanding where is large number of fraudster and what type of transaction is done by them
	2. Data ANALYSIS ->
		Feature Extraction is done as explained above
	3. MODELLING ->
	1. Using Oversampling - detection of fraud (recall) is increased
	Without oversampling classification report 
			1 mean Fraudster
			0 mean No Fraudster
				precision    recall  f1-score   support

			0       0.99      1.00      0.99    124324
			1       0.86      0.64      0.73      2709

		accuracy                           0.99    127033
	macro avg       0.92      0.82      0.86    127033
	weighted avg       0.99      0.99      0.99    127033

			Confusion matrix
			[[124031    293]
				[   975   1734]]

	2. With oversampling using SMOTE - 
			1 mean Fraudster
			0 mean No Fraudster
				Classification Report 
				precision    recall  f1-score   support

			0       1.00      0.98      0.99    124324
			1       0.50      0.83      0.62      2709

		accuracy                           0.98    127033
	macro avg       0.75      0.90      0.81    127033
	weighted avg       0.99      0.98      0.98    127033

				Confusion matrix
				[[122089   2235]
				[   468   2241]]

	4. ACTION -> After finding transacation which is fraud try to take action on user who did that transaction
	1. Action can be to block or locked user
	2. Action can be to increase flag for that user by 1 and if increase by a thershold may be 3 block that user
	3. Implement more advanced model which will take time to finalise if that user past activity is like a fraud for that
	we can implment fraud detection for a user profile taking features like number of transaction in past, how frequently with what amount and recency and so on and if that user if found fraud by this model then lock it.
	4. This have to be done so as to put checks for our model and don't lock user for any fraud transaction which can not be due to model error. This will help in keeping good brand of company Revoult by giving good customer service.
	5. We can also do second check by using agent which will validate if transaction is fraud or not.
	6. This can be implment at initial stage so as to make a feedback loop for our model and it helps in increasing dataset for modeling


