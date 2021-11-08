# Load the necessarty libraries
from mlxtend.preprocessing import TransactionEncoder
import pandas

Imaginary_Store = pandas.read_csv('C:\\IIT\\Machine Learning\\Data\\Imaginary_Store.csv',
                                  delimiter=',')

# Convert the Sale Receipt data to the Item List format
ListItem = Imaginary_Store.groupby(['Customer'])['Item'].apply(list).values.tolist()

# Convert the Item List format to the Item Indicator format
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)

# Calculate the frequency table of number of customers per item
nCustomerPurchase = Imaginary_Store.groupby('Item').size()
freqTable = pandas.Series.sort_index(pandas.Series.value_counts(nCustomerPurchase))
print('Frequency of Number of Customers Purchase Item')
print(freqTable)

# Calculate the frequency table of number of items purchase
nItemPurchase = Imaginary_Store.groupby('Customer').size()
freqTable = pandas.Series.sort_index(pandas.Series.value_counts(nItemPurchase))
print('Frequency of Number of Items Purchase')
print(freqTable)

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Find the frequent itemsets
frequent_itemsets = apriori(ItemIndicator, min_support = 0.01, max_len = 7, use_colnames = True)

# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.5)

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

# Find the frequent itemsets
frequent_itemsets = apriori(ItemIndicator, min_support = 0.1, max_len = 7, use_colnames = True)

# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.8)

assoc_rules['lift'].describe()

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

# Show rules that have the 'CEREAL' consquent
import numpy
Cereal_Consequent_Rule = assoc_rules[numpy.isin(assoc_rules["consequents"].values, {"Cereal"})]

# Show rules that have the 'Oranges' antecedent
antecedent = assoc_rules["antecedents"]
selectAntecedent = numpy.ones((assoc_rules.shape[0], 1), dtype=bool)

i = 0
for f in antecedent:
    selectAntecedent[i,0] = "Oranges" in f
    i = i + 1
  
Orange_Antecedent_Rule = assoc_rules[selectAntecedent]
