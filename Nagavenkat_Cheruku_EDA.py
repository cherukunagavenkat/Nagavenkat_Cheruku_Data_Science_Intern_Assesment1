import pandas as pd

customer_dataFile = pd.read_csv(r"Customers.csv")
products_dataFile = pd.read_csv(r"Products.csv")
transactions_dataFile = pd.read_csv(r"Transactions.csv")

customer_dataFile.info()

# Here in customer data set there are no null values.
# But SignupDate column is object data type. So, change it to date datatype.

customer_dataFile['SignupDate'] = pd.to_datetime(customer_dataFile['SignupDate'], format='%Y-%m-%d').dt.date

customer_dataFile.describe()


products_dataFile.info()

# Everything looks good


transactions_dataFile.info()

transactions_dataFile['TransactionDate'] = pd.to_datetime(transactions_dataFile['TransactionDate'], errors='coerce')

##############################################


merged_df = transactions_dataFile.merge(products_dataFile, on='ProductID', how='left') \
                        .merge(customer_dataFile, on='CustomerID', how='left')

merged_df.to_csv('merged_data.csv', index=False)

merged_df.info()


### First Business Moment Decision ###

# Mean
print("Total Value mean value is: ",merged_df.TotalValue.mean())
print("Price_x mean value is: ",merged_df.Price_x.mean())
print("Price_y mean value is: ",merged_df.Price_y.mean())

# Median 
print("Total Value median value is: ",merged_df.TotalValue.median())
print("Price_x median value is: ",merged_df.Price_x.median())
print("Price_y median value is: ",merged_df.Price_y.median())

'''
The mean value of Total Value is 689.99 USD
The median value of Total Value is 588.88 USD

Total Value has a mean > median, which is right-skewed distribution, it means there are higher outliers.

The mean value of Price_x and Price_y is 272.55407 USD
The median value of Price_x and Price_y is 299.93 USD

Price_x and Price_y have left-skewed distributions as mean < median, suggesting lower outliers.

'''

# Mode
merged_df['ProductName'].value_counts()
merged_df['Category'].value_counts()

print("Mode of ProductName is: ",merged_df['ProductName'].mode().tolist())

print("Mode of category is: ",merged_df['Category'].mode().tolist())

### Second business moment ###

#Varience 
print("Varience of TotalValue is: ",merged_df.TotalValue.var())
print("Price_x Varience value is: ",merged_df.Price_x.var())
print("Price_y Varience value is: ",merged_df.Price_y.var())

# Standard Deviation
print("Standard deviation of TotalValue is: ",merged_df.TotalValue.std())
print("Price_x Standard deviation value is: ",merged_df.Price_x.std())
print("Price_y Standard deviation value is: ",merged_df.Price_y.std())

#Range
print("The range of Tatal Value is : ",max(merged_df.TotalValue) - min(merged_df.TotalValue))
print("The range of Price_x and Price_y is : ",max(merged_df.Price_x) - min(merged_df.Price_x))

'''
The varience and standard deviation of Total Value 243191.47 USD and 493.14 USD respectively shows significant variation in transaction amounts,
indicating diverse purchasing behaviours and product combinations.

The varience and standard deviation of Price_x and Price_y are 19806.73 USD and 140.73 USD respectively shows the wide range of product pricing.

The range of total value is 1974.96 USD and the range of Price_x and Price_y is 481.68 USD.
'''

# Third Business moment decision

print("The skewness value of Total Value is: ",merged_df.TotalValue.skew())
print("The skewness value of Price_x is: ",merged_df.Price_x.skew())
print("The skewness value of Price_y is ",merged_df.Price_y.skew())

# Fourth business moment decision

print("The kurtosis of Total Value is: ",merged_df.TotalValue.kurt())
print("The kurtosis of Price_x is: ", merged_df.Price_x.kurt())
print("The kurtosis of Price_y is: ", merged_df.Price_y.kurt())

'''
Total Value is positively skewed, meaning potential high outliers.
Price_x and Price_y are slightly negatively skewed, meaning potential low outliers.

All distributions are platykurtic, meaning the data has fewer outliers and is less concentrated around the mean.
'''

#####################################################################################################################

### Data Visualization ###############

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Exploratory Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(data=merged_df, x='Region', order=merged_df['Region'].value_counts().index)
plt.title('Number of Customers by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=merged_df, x='Category', y='Price_x', ci=None, estimator=sum)
plt.title('Total Revenue by Product Category')
plt.xlabel('Category')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.show()

merged_df.groupby('YearMonth')['TotalValue'].sum().plot()

# Time-based Analysis
merged_df['YearMonth'] = pd.to_datetime(merged_df['TransactionDate'], errors='coerce').dt.to_period('M')

plt.figure(figsize=(12, 6))
merged_df.groupby('YearMonth')['TotalValue'].sum().plot()
plt.title('Monthly Revenue Trend')
plt.xlabel('Year-Month')
plt.ylabel('Total Revenue')
plt.grid()
plt.show()

'''

##### Insights #########

1. From the Region vs Count plot it is very clear that, South America has more customer count and Asia had the least customer count
2. From Category vs Total Revenue plot it is very clear that, Books category has more total revenue and Home decor has the less total revenue comparitvely
3. From Monthly Revenue Trend it is clear that, July had the high monthly revenue and November had the least monthly revenue
4. The average purchase value per transaction is 689.99 USD
5. Mostly purchased Product is ActiveWear Smartwatch.

''' 

insights = [
    "1. From the Region vs Count plot it is very clear that, South America has more customer count and Asia had the least customer count",
    "2. From Category vs Total Revenue plot it is very clear that, Books category has more total revenue and Home decor has the less total revenue comparitvely",
    "3. From Monthly Revenue Trend it is clear that, July had the high monthly revenue and November had the least monthly revenue",
    "4. The average purchase value per transaction is 689.99 USD",
    "5. Mostly purchased Product is ActiveWear Smartwatch."
]

text_file_path = r'D:\Job\EDA_Insights_Report.txt'

# Write insights to the text file
with open(text_file_path, 'w') as file:
    for insight in insights:
        file.write(insight + '\n')

print(f"Insights saved to {text_file_path}")
