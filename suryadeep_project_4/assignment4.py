import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Module to interpolate values.
def interpolate_missing_vals(d_frame, i):
    s1 = []
    store_val = []
    x = 0
    for elems in d_frame.iloc[i]:
        s1.append(elems)
        
    data = {'vals' : s1}
    df = pd.DataFrame(data = data)

    # Interpolate the values.
    df['vals'].interpolate(method = 'pad', limit = 2, inplace = True)
    store_val = list(df['vals'])
    
    # Append to the data frame
    for cols in cgm_val_df.columns:
        cgm_val_df.at[i, cols] = store_val[x]
        x += 1

def create_bins(df_cb, df_cb_bin):
    # Create bin values.
    for cols in df_cb.columns:
        for i in range(0, len(df_cb)):
            residue = df_cb[cols][i] % 10
            low = df_cb[cols][i] - residue
            high = low + 9
            df_cb_bin[cols][i] = str(low) + '-' + str(high)
    return df_cb_bin

def find_cgm_max(df_cb, df_cb_bin, mx_index_lst):
    max_val_lst = []
    max_lst = []

    # Create a maximum value list.
    indx = 0
    # FInd max values per row.
    for i in range(0, len(df_cb)):
        max_lst.append(df_cb.loc[i].idxmax())

    for elems in mx_index_lst:
        max_val_lst.append(df_cb_bin[elems][indx])
        indx += 1
    return max_val_lst
    
def sixth_cgm_val(df_cb, df_cb_bin):
    max_ix_lst = []
    max_sixth_lst = []
    for i in range(0, len(df_cb)):
        max_ix_lst.append(df_cb.loc[i].idxmax())
        max_sixth_lst.append(df_cb_bin[df_cb_bin.columns[-6]][i])
    return max_ix_lst, max_sixth_lst

def create_ib_vals(ival_df, itime_df, l_time_df):
    store_ib = []
    # Calculate Ib value.
    # Find non zero values.
    for i in range(0, len(ival_df)):
        local_val_store = []
        local_index_store = []
        local_time_diff = []

        min_idx = -1
        col_cntr = 0
        sixth_date  = l_time_df[l_time_df.columns[-6]][i]
        for cols in ival_df.columns:
            if ival_df[cols][i] > 0:
                local_val_store.append(ival_df[cols][i])
                local_index_store.append(col_cntr)
            col_cntr += 1
        
        # Find date of the non zero values.
        if len(local_index_store) == 1:
            store_ib.append(local_val_store[0])
        else:
            for idx in local_index_store:
                local_time_diff.append(abs(sixth_date - itime_df[itime_df.columns[idx]][i]))
            min_idx = local_time_diff.index(min(local_time_diff))
            store_ib.append(local_val_store[min_idx])
    return store_ib

def preform_apriori(assoc_df):
    # Find frequent sets
    interim_vals = []
    for i in range(assoc_df.shape[0]):
        interim_vals.append([str(assoc_df.values[i,j]) for j in range(0,3)])

    te = TransactionEncoder()
    encode_df = pd.DataFrame(te.fit(interim_vals).transform(interim_vals), columns = te.columns_)
    #d = pd.DataFrame(te_ary, columns = te.columns_)

    frequent_itemsets = apriori(encode_df, min_support=0.001, use_colnames=True)
    association_op = association_rules(frequent_itemsets, metric="support", min_threshold=0.001)

    association_op["_size"] = association_op["antecedents"].apply(lambda x: len(x))
    association_op = association_op[(association_op['_size'] >= 2)]
    association_op['antecedents'] = association_op['antecedents'].astype(str)
    association_op['consequents'] = association_op['consequents'].astype(str)
    association_op = association_op[~association_op["antecedents"].str.contains("I_B")]
    association_op = association_op[~association_op["consequents"].str.contains("CGM_M")]
    association_op = association_op[~association_op["consequents"].str.contains("CGM_0")]

    for i in range(len(association_op)):
        association_op['antecedents'].iloc[i] = association_op['antecedents'].iloc[i].replace('frozenset','')
        association_op['consequents'].iloc[i] = association_op['consequents'].iloc[i].replace('frozenset','')
        association_op['antecedents'].iloc[i] = association_op['antecedents'].iloc[i].replace('({','')
        association_op['antecedents'].iloc[i] = association_op['antecedents'].iloc[i].replace('})','')
        association_op['consequents'].iloc[i] = association_op['consequents'].iloc[i].replace('({','')
        association_op['consequents'].iloc[i] = association_op['consequents'].iloc[i].replace('})','')
        association_op['antecedents'].iloc[i] = association_op['antecedents'].iloc[i].replace("'",'')
        association_op['consequents'].iloc[i] = association_op['consequents'].iloc[i].replace("'",'')
    association_op = association_op.reset_index(drop = True)
    return frequent_itemsets, association_op


'''
    Code to trigger the functions
    -----------------------------
'''

# Read for patient 3.
lunch_time_df_3 = pd.read_csv('Data/CGMDatenumLunchPat3.csv')
cgm_val_df_3 = pd.read_csv('Data/CGMSeriesLunchPat3.csv')
insulin_time_df_3 = pd.read_csv('Data/InsulinDatenumLunchPat3.csv')
insulin_val_df_3 = pd.read_csv('Data/InsulinBolusLunchPat3.csv')

# Read for patient 4.
lunch_time_df_4 = pd.read_csv('Data/CGMDatenumLunchPat4.csv')
cgm_val_df_4 = pd.read_csv('Data/CGMSeriesLunchPat4.csv')
insulin_time_df_4 = pd.read_csv('Data/InsulinDatenumLunchPat4.csv')
insulin_val_df_4 = pd.read_csv('Data/InsulinBolusLunchPat4.csv')

# Read for patient 5.
lunch_time_df_5 = pd.read_csv('Data/CGMDatenumLunchPat5.csv')
cgm_val_df_5 = pd.read_csv('Data/CGMSeriesLunchPat5.csv')
insulin_time_df_5 = pd.read_csv('Data/InsulinDatenumLunchPat5.csv')
insulin_val_df_5 = pd.read_csv('Data/InsulinBolusLunchPat5.csv')


'''
    Patient-1
    --------
'''
# Read for patient 1.
lunch_time_df_1 = pd.read_csv('Data/CGMDatenumLunchPat1.csv')
cgm_val_df_1 = pd.read_csv('Data/CGMSeriesLunchPat1.csv')
insulin_time_df_1 = pd.read_csv('Data/InsulinDatenumLunchPat1.csv')
insulin_val_df_1 = pd.read_csv('Data/InsulinBolusLunchPat1.csv')

        
lunch_time_df_1.fillna(0, inplace = True)
insulin_time_df_1.fillna(0, inplace = True)
#cgm_val_df_1.dropna(inplace = True)
cgm_val_df_1.fillna(0, inplace = True)
cgm_val_df_1 = cgm_val_df_1.astype(int)

insulin_val_df_1.fillna(0, inplace = True)

bin_cgm_val_df_1 = cgm_val_df_1.copy()
bin_cgm_val_df_1 = bin_cgm_val_df_1.reset_index(drop = True)
cgm_val_df_1 = cgm_val_df_1.reset_index(drop = True)
bin_cgm_val_df_1 = create_bins(cgm_val_df_1, bin_cgm_val_df_1)

# Create Dataframe for association.
assoc_df_1 = pd.DataFrame(columns = ['CGM_M', 'CGM_0', 'I_B'])

# Get the maximum indexes.
max_index_list_1, max_sixth_lst_1 = sixth_cgm_val(cgm_val_df_1, bin_cgm_val_df_1)
max_val_lst_1 = find_cgm_max(cgm_val_df_1, bin_cgm_val_df_1, max_index_list_1)
store_ib_1 = create_ib_vals(insulin_val_df_1, insulin_time_df_1, lunch_time_df_1)
print(max_val_lst_1)
# Add values in the dataframe.
assoc_df_1['CGM_M'] = max_val_lst_1
assoc_df_1['CGM_0'] = max_sixth_lst_1
assoc_df_1['I_B'] = store_ib_1

for i in range(0, len(assoc_df)):
    assoc_df_1['CGM_M'][i] = 'CGM_M-' + assoc_df_1['CGM_M'][i]
    assoc_df_1['CGM_0'][i] = 'CGM_0-' + assoc_df_1['CGM_0'][i]
    assoc_df_1['I_B'][i] = 'I_B-' + str(assoc_df_1['I_B'][i])

fi1, ar1 = preform_apriori(assoc_df_1)
fi1 = fi1['itemsets']
'''
    Patient-2
    --------
'''
# Read for patient 2.
lunch_time_df_2 = pd.read_csv('Data/CGMDatenumLunchPat2.csv')
cgm_val_df_2 = pd.read_csv('Data/CGMSeriesLunchPat2.csv')
insulin_time_df_2 = pd.read_csv('Data/InsulinDatenumLunchPat2.csv')
insulin_val_df_2 = pd.read_csv('Data/InsulinBolusLunchPat2.csv')

        
lunch_time_df_2.fillna(0, inplace = True)
insulin_time_df_2.fillna(0, inplace = True)
cgm_val_df_2.fillna(0, inplace = True)
cgm_val_df_2 = cgm_val_df_2.astype(int)

insulin_val_df_2.fillna(0, inplace = True)

bin_cgm_val_df_2 = cgm_val_df_2.copy()
bin_cgm_val_df_2 = bin_cgm_val_df_2.reset_index(drop = True)
cgm_val_df_2 = cgm_val_df_2.reset_index(drop = True)
bin_cgm_val_df_2 = create_bins(cgm_val_df_2, bin_cgm_val_df_2)

# Create Dataframe for association.
assoc_df_2 = pd.DataFrame(columns = ['CGM_M', 'CGM_0', 'I_B'])

# Get the maximum indexes.
max_index_list_2, max_sixth_lst_2 = sixth_cgm_val(cgm_val_df_2, bin_cgm_val_df_2)
max_val_lst_2 = find_cgm_max(cgm_val_df_2, bin_cgm_val_df_2, max_index_list_2)
store_ib_2 = create_ib_vals(insulin_val_df_2, insulin_time_df_2, lunch_time_df_2)
print(max_val_lst_2)
# Add values in the dataframe.
assoc_df_2['CGM_M'] = max_val_lst_2
assoc_df_2['CGM_0'] = max_sixth_lst_2
assoc_df_2['I_B'] = store_ib_2

for i in range(0, len(assoc_df_2)):
    assoc_df_2['CGM_M'][i] = 'CGM_M-' + assoc_df_2['CGM_M'][i]
    assoc_df_2['CGM_0'][i] = 'CGM_0-' + assoc_df_2['CGM_0'][i]
    assoc_df_2['I_B'][i] = 'I_B-' + str(assoc_df_2['I_B'][i])

fi2, ar2 = preform_apriori(assoc_df_2)
fi2 = fi2['itemsets']

'''
    Patient-3
    --------
'''
# Read for patient 3.
lunch_time_df_3 = pd.read_csv('Data/CGMDatenumLunchPat3.csv')
cgm_val_df_3 = pd.read_csv('Data/CGMSeriesLunchPat3.csv')
insulin_time_df_3 = pd.read_csv('Data/InsulinDatenumLunchPat3.csv')
insulin_val_df_3 = pd.read_csv('Data/InsulinBolusLunchPat3.csv')
        
lunch_time_df_3.fillna(0, inplace = True)
insulin_time_df_3.fillna(0, inplace = True)
cgm_val_df_3.fillna(0, inplace = True)
cgm_val_df_3 = cgm_val_df_3.astype(int)

insulin_val_df_3.fillna(0, inplace = True)

bin_cgm_val_df_3 = cgm_val_df_3.copy()
bin_cgm_val_df_3 = bin_cgm_val_df_3.reset_index(drop = True)
cgm_val_df_3 = cgm_val_df_3.reset_index(drop = True)
bin_cgm_val_df_3 = create_bins(cgm_val_df_3, bin_cgm_val_df_3)

# Create Dataframe for association.
assoc_df_3 = pd.DataFrame(columns = ['CGM_M', 'CGM_0', 'I_B'])

# Get the maximum indexes.
max_index_list_3, max_sixth_lst_3 = sixth_cgm_val(cgm_val_df_3, bin_cgm_val_df_3)
max_val_lst_3 = find_cgm_max(cgm_val_df_3, bin_cgm_val_df_3, max_index_list_3)
store_ib_3 = create_ib_vals(insulin_val_df_3, insulin_time_df_3, lunch_time_df_3)
print(max_val_lst_3)
# Add values in the dataframe.
assoc_df_3['CGM_M'] = max_val_lst_3
assoc_df_3['CGM_0'] = max_sixth_lst_3
assoc_df_3['I_B'] = store_ib_3

for i in range(0, len(assoc_df_3)):
    assoc_df_3['CGM_M'][i] = 'CGM_M-' + assoc_df_3['CGM_M'][i]
    assoc_df_3['CGM_0'][i] = 'CGM_0-' + assoc_df_3['CGM_0'][i]
    assoc_df_3['I_B'][i] = 'I_B-' + str(assoc_df_3['I_B'][i])

fi3, ar3 = preform_apriori(assoc_df_3)
fi3 = fi3['itemsets']

'''
    Patient-4
    --------
'''
# Read for patient 4.
lunch_time_df_4 = pd.read_csv('Data/CGMDatenumLunchPat4.csv')
cgm_val_df_4 = pd.read_csv('Data/CGMSeriesLunchPat4.csv')
insulin_time_df_4 = pd.read_csv('Data/InsulinDatenumLunchPat4.csv')
insulin_val_df_4 = pd.read_csv('Data/InsulinBolusLunchPat4.csv')
        
lunch_time_df_4.fillna(0, inplace = True)
insulin_time_df_4.fillna(0, inplace = True)
cgm_val_df_4.fillna(0, inplace = True)
cgm_val_df_4 = cgm_val_df_4.astype(int)

insulin_val_df_4.fillna(0, inplace = True)

bin_cgm_val_df_4 = cgm_val_df_4.copy()
bin_cgm_val_df_4 = bin_cgm_val_df_4.reset_index(drop = True)
cgm_val_df_4 = cgm_val_df_4.reset_index(drop = True)
bin_cgm_val_df_4 = create_bins(cgm_val_df_4, bin_cgm_val_df_4)

# Create Dataframe for association.
assoc_df_4 = pd.DataFrame(columns = ['CGM_M', 'CGM_0', 'I_B'])

# Get the maximum indexes.
max_index_list_4, max_sixth_lst_4 = sixth_cgm_val(cgm_val_df_4, bin_cgm_val_df_4)
max_val_lst_4 = find_cgm_max(cgm_val_df_4, bin_cgm_val_df_4, max_index_list_4)
store_ib_4 = create_ib_vals(insulin_val_df_4, insulin_time_df_4, lunch_time_df_4)
print(max_val_lst_4)
# Add values in the dataframe.
assoc_df_4['CGM_M'] = max_val_lst_4
assoc_df_4['CGM_0'] = max_sixth_lst_4
assoc_df_4['I_B'] = store_ib_4

for i in range(0, len(assoc_df_4)):
    assoc_df_4['CGM_M'][i] = 'CGM_M-' + assoc_df_4['CGM_M'][i]
    assoc_df_4['CGM_0'][i] = 'CGM_0-' + assoc_df_4['CGM_0'][i]
    assoc_df_4['I_B'][i] = 'I_B-' + str(assoc_df_4['I_B'][i])

fi4, ar4 = preform_apriori(assoc_df_4)
fi4 = fi4['itemsets']
'''
    Patient-5
    --------
'''
# Read for patient 5.
lunch_time_df_5 = pd.read_csv('Data/CGMDatenumLunchPat5.csv')
cgm_val_df_5 = pd.read_csv('Data/CGMSeriesLunchPat5.csv')
insulin_time_df_5 = pd.read_csv('Data/InsulinDatenumLunchPat5.csv')
insulin_val_df_5 = pd.read_csv('Data/InsulinBolusLunchPat5.csv')
        
lunch_time_df_5.fillna(0, inplace = True)
insulin_time_df_5.fillna(0, inplace = True)
cgm_val_df_5.fillna(0, inplace = True)
cgm_val_df_5 = cgm_val_df_5.astype(int)

insulin_val_df_5.fillna(0, inplace = True)

bin_cgm_val_df_5 = cgm_val_df_5.copy()
bin_cgm_val_df_5 = bin_cgm_val_df_5.reset_index(drop = True)
cgm_val_df_5 = cgm_val_df_5.reset_index(drop = True)
bin_cgm_val_df_5 = create_bins(cgm_val_df_5, bin_cgm_val_df_5)

# Create Dataframe for association.
assoc_df_5 = pd.DataFrame(columns = ['CGM_M', 'CGM_0', 'I_B'])

# Get the maximum indexes.
max_index_list_5, max_sixth_lst_5 = sixth_cgm_val(cgm_val_df_5, bin_cgm_val_df_5)
max_val_lst_5 = find_cgm_max(cgm_val_df_5, bin_cgm_val_df_5, max_index_list_5)
store_ib_5 = create_ib_vals(insulin_val_df_5, insulin_time_df_5, lunch_time_df_5)
print(max_val_lst_5)
# Add values in the dataframe.
assoc_df_5['CGM_M'] = max_val_lst_5
assoc_df_5['CGM_0'] = max_sixth_lst_5
assoc_df_5['I_B'] = store_ib_5
assoc_df_5 = assoc_df_5.reset_index(drop = True)
for i in range(0, len(assoc_df_5)):
    assoc_df_5['CGM_M'][i] = 'CGM_M-' + assoc_df_5['CGM_M'][i]
    assoc_df_5['CGM_0'][i] = 'CGM_0-' + assoc_df_5['CGM_0'][i]
    assoc_df_5['I_B'][i] = 'I_B-' + str(assoc_df_5['I_B'][i])

fi5, ar5 = preform_apriori(assoc_df_5)
fi5 = fi5['itemsets']
# Create csv for item set
conf_df = pd.concat([fi1, fi2, fi3, fi4, fi5], ignore_index=True)
conf_df.to_csv('itemsets.csv', index=False)

# Create highest value csv
ar_conf_df = pd.concat([ar1, ar2, ar3, ar4, ar5], ignore_index=True)
interim_vals = list(set(ar_conf_df['confidence']))
second_last = sorted(interim_vals)[-2]
highest_csv = ar_conf_df.loc[ar_conf_df['confidence'] <= second_last]
highest_csv = highest_csv['antecedents']
highest_csv.to_csv('highest_vals.csv', index=False)

# Create anomalous value csv
anonal_csv = ar_conf_df.loc[ar_conf_df['confidence'] <= .15]
anonal_csv = anonal_csv['antecedents']
anonal_csv.to_csv('anomalous.csv', index=False)