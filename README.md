import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Dataset i have used in order to explain the code
dataset = [
    ['Milk', 'Bread', 'Eggs'],
    ['Bread', 'Diapers', 'Beer', 'Eggs'],
    ['Milk', 'Diapers', 'Beer', 'Cola'],
    ['Bread', 'Milk', 'Diapers', 'Beer'],
    ['Bread', 'Milk', 'Diapers', 'Cola']
]

#Transactions done
te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_array, columns=te.columns_)

# Appylying algorithm 
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

#Asoociation Rules 
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

#Results
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


#If we add a data set to this the model would work in a similar manner because of the algorithm used and the association rules .
