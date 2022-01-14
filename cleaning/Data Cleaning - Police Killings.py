# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nameparser import HumanName
import webbrowser
# %matplotlib inline

# %% [markdown]
# Import CSV file of data that needs cleaning and check its shape.

# %%
killings = pd.read_csv('./csv_files/police_killings_original.csv')
killings.shape

# %% [markdown]
# # Data Cleaning
#
# Drop rows that contain ALL null values, then drop columns that contain ALL null values

# %%
killings.dropna(how='all', axis=0, inplace=True)
killings.dropna(how='all', axis=1, inplace=True)
killings.shape

# %%
killings.columns

# %% [markdown]
# ## Cleaning column names and dropping unnecessary columns
# The column names are pretty messy, let's change that

# %%
killings.rename(columns={
    "Victim's name":"victims_name", 
    "Victim's age":"victims_age", 
    "Victim's gender":"victims_gender", 
    "Victim's race":"victims_race",
    "URL of image of victim": "victim_img_url",
    "Date of Incident (month/day/year)":"date",
    "Street Address of Incident": "street_address",
    "City":"city",
    "State":"state",
    "Zipcode":"zipcode",
    "County":"county",
    "Agency responsible for death":"agency_resp_for_death",
    "Cause of death":"cause_of_death",
    "A brief description of the circumstances surrounding the death":"desc_of_circumstances",
    "Official disposition of death (justified or other)":"official_disposition_of_death",
    "Criminal Charges?":"criminal_charges", 
    'Link to news article or photo of official document':"news_article_link",
    'Symptoms of mental illness?':'mental_illness',
    "Unarmed":"unarmed",
    'Alleged Weapon (Source: WaPo)':'alleged_weapon',
    'Alleged Threat Level (Source: WaPo)':'threat_level',
    'Fleeing (Source: WaPo)': 'fleeing',
    'Body Camera (Source: WaPo)':'body_camera',
    'WaPo ID (If included in WaPo database)':'WaPo_id',
    'Off-Duty Killing?':'off_duty_killing',
    'Geography (via Trulia methodology based on zipcode population density: http://jedkolko.com/wp-content/uploads/2015/05/full-ZCTA-urban-suburban-rural-classification.xlsx )':'geo_type',
    }, inplace=True)

# %% [markdown]
# Check to see what percentage of null values are contained in each column that remains.

# %%
(killings.isnull().sum() / len(killings)) * 100

# %% [markdown]
# Because off_duty_killing is missing 97% of values, I'm going to drop it.
#
# Because we can use our df index to ID each row, we don't need the ID column and can drop it.
#
# I've also tried to access the WaPo database to look up cases based on WaPo_id, but it doesn't let me search using that parameter, so I'm going to drop that column too.

# %%
killings.drop(['WaPo_id', 'off_duty_killing', 'ID'], axis=1, inplace=True)

# %% [markdown]
# Let's investigate what our columns are about and perform any necessary cleaning column by column

# %%
killings.info()

# %% [markdown]
# ## Converting data types

# %% [markdown]
# ### Date Column

# %% [markdown]
# We can convert the dates in the date column to Python datetime objects

# %%
killings["date"] = pd.to_datetime(killings["date"], infer_datetime_format=True)

# %%

# %%

# %%

# %% [markdown] tags=[]
# ## Handling Null values

# %% [markdown]
# ### URL column

# %% [markdown]
# For rows that don't have a URL image of the victim, I'm going to impute "None"

# %%
killings['victim_img_url'].fillna('None', inplace=True)

# %% [markdown]
# ### Gender
# Let's try to determine what to do with the Null values for gender.  We'll begin by inspecting those that have a news article link present as that may have information we can use.

# %%
null_gender_df = killings[(killings["victims_gender"].isnull()==True) & (killings['news_article_link'].isnull()==False)]

null_gender_df

# %% [markdown]
# While reading through each article, it would be good to know which columns are Null in case we find that information in the article.  To do this, we'll determine which columns are null for each of the distinct rows, then read the article and impute any information we find.

# %%
null_gender_links = null_gender_df['news_article_link']

null_gender_idx = null_gender_links.index
null_gender_urls = null_gender_links.to_list()

print("Here are the links for news articles that contain links for rows with a null gender:")
print("-"*50)
print()
for idx, link in zip(null_gender_idx, null_gender_links):
    null_col_mask = killings.loc[idx].isna()==True
    print(F"df_idx: {idx}")
    print(killings.loc[idx, null_col_mask].to_string())
    print(F"link: {link}")
    print()

# %%
# news article says the victim was male, no age given
killings.loc[13, 'victims_gender'] = 'Male'

# %%
# news article says the victim was male in his 40s
killings.loc[112, ['victims_gender', 'victims_age']] = 'Male', '40'

# %%
# URL mentions male victim, but ad is behind paywall so can't investigate further
killings.loc[1029, 'victims_gender'] = 'Male'

# %%
killings.loc[774]

# %%
killings[(killings['victims_gender'].isnull()==True) & (killings['news_article_link'].isnull()==True)]

# %%
# Both names sound male, so I'll set the gender accordingly
killings.loc[528, 'victims_gender'] = 'Male'
killings.loc[774, 'victims_gender'] = 'Male'

# %%
num_null_gender = killings['victims_gender'].isnull().sum()
print(F"There are only {num_null_gender} rows with a null gender left, so we'll impute 'Unknown'")

killings['victims_gender'].fillna('Unknown', inplace=True)

# %%

# %%

# %% [markdown]
# ### City/County

# %%
killings.loc[killings['city'].isnull()==True,['city', 'state', 'zipcode']]

# %% [markdown]
# Since we have a zipcode and state for several of these rows, we can look up the appropriate city

# %%
killings.loc[3339, 'city'] = "Land O' Lakes"
killings.loc[5561, 'city'] = "Jacksonville"
killings.loc[6511, 'city'] = "Douglas"

# %% [markdown]
# #### not sure if I should keep this - BEGIN

# %%
null_city_df = killings.loc[killings['city'].isnull()==True,:]

null_city_df

# %%
null_city_links = null_city_df['news_article_link']

null_city_idx = null_city_links.index
null_city_urls = null_city_links.to_list()

print("Here are the links for news articles that contain links for rows with a null city:")
print("-"*50)
print()
for idx, link in zip(null_city_idx[0:10], null_city_links[0:10]):
    null_col_mask = killings.loc[idx].isna()==True
    print(F"df_idx: {idx}")
    print(killings.loc[idx, null_col_mask].to_string())
    print(F"link: {link}")
    print()

# %% [markdown]
# #### not sure if I should keep this END

# %%
killings.isnull().sum()

# %%
killings.loc[killings['county'].isnull()==True,['street_address', 'city', 'state', 'zipcode', 'county']]

# %% [markdown]
# Since we have city and state information for these rows, we can look up the appropriate county

# %%
killings.loc[493,'county'] = 'Copiah'
killings.loc[528,'county'] = 'Wyandotte'
killings.loc[774,'county'] = 'Genesee'
killings.loc[1250,'county'] = 'Pratt'
killings.loc[1305,'county'] = 'Gadsden'
killings.loc[1322,'county'] = 'Hunt'
killings.loc[1336,'county'] = 'Utah'
killings.loc[1346,'county'] = 'Milwaukee'
killings.loc[1356,'county'] = 'Pemiscot'
killings.loc[1367,'county'] = 'Loudon'
killings.loc[1430,'county'] = 'Pierce'
killings.loc[1607,'county'] = 'Caldwell'
killings.loc[1965,'county'] = 'Maricopa'
killings.loc[1981,'county'] = 'Daviess'
killings.loc[3315,'county'] = 'Lake'

# %% [markdown]
# ### Description of Circumstances Surrounding Death
# It'll take too much time to research these missing entries and I don't think they'll contribute much to analysis, so I'll fill them with "Unavailable"

# %%
killings['desc_of_circumstances'].fillna('Unavailable', inplace=True)

# %% [markdown]
# ### News Article Link
# It's going to be near impossible to research links for these rows, so we'll also fill them with "Unavailable"

# %%
killings['news_article_link'].fillna('Unavailable', inplace=True)

# %% [markdown] tags=[]
# ### Symptoms of Mental Illness

# %%
killings['mental_illness'].unique()

# %% [markdown]
# Because "Unknown", "Unkown", "Unknown ", and "unknown" all represent the same idea, we need to distill them down to a single category.

# %%
mental_illness_dict = {"Unkown":"unknown",
                       "Unknown ":"unknown",
                       "Unknown":"unknown"}

killings['mental_illness'] = killings['mental_illness'].map(mental_illness_dict).fillna(killings['mental_illness'])

# %% [markdown]
# ### Geography Type/Zipcode/Street Address

# %%
killings['geo_type'].unique()

# %% [markdown]
# We'll start by looking at placed that have a street address listed so we can use Google maps and try to fill in missing data in these columns

# %%
killings.loc[(killings['geo_type'].isnull()==True) & (killings['street_address'] != 'None'), ['street_address', 'city', 'state', 'zipcode', 'geo_type']]

# %% [markdown]
# Time to update the null values in the geo_type column.  Note: The data in the data set uses households/sq mi based on zip codes to establish urban, suburban, or rural classification, but for the null values I used households/sq mi based on city instead as the information was more readily available.
#
# - urban: households per square mile >=2213.2 
# - suburban: households per square mile >=101.6 and < 2213.2
# - rural: households per square mile <101.6
#
# [Source for geo_type classifications](http://jedkolko.com/wp-content/uploads/2015/05/Data-and-methodological-details-052715.pdf)

# %%
killings.loc[522, 'geo_type'] = 'Suburban' #82540 households / 62.42 sq mi = 1322 households / sq mi (suburban)

# %%
killings.loc[595, 'geo_type'] = 'Suburban' #521198 households / 385.8 sq mi = 1351 households / sq mi (suburban)

# %%
killings.loc[1000, 'geo_type'] = 'Rural'

# %%
killings.loc[1004, 'geo_type'] = 'Rural'

# %%
killings.loc[1281, 'geo_type'] = 'Suburban' # 156,482 households / 156.6 sq. mi = 999 households / sq mi (suburban)

# %%
killings.loc[1947, 'geo_type'] = 'Suburban' # 48095 households / 108.3 sq. mi = 444 households / sq mi (suburban)

# %%
killings.loc[2072, 'geo_type'] = 'Suburban' # 33 households / .22 sq. mi = 150 households / sq mi (suburban, but just barely)

# %%
killings.loc[2207, 'geo_type'] = 'Suburban' # 321835 households / 181.4 sq. mi = 1774 households / sq mi (suburban, almost urban)

# %%
killings.loc[2419, 'geo_type'] = 'Urban' # 130885 households / 22.78 sq. mi = 5745 households / sq mi (urban)

# %%
killings.loc[2488, 'geo_type'] = 'Suburban' # 23 households / .14 sq. mi = 164 households / sq mi (Suburban, just barely)

# %%
killings.loc[3315, 'geo_type'] = 'Suburban' # 7013 households / 5.598 sq. mi = 1263 households / sq mi (suburban)

# %%
killings.loc[3347, 'geo_type'] = 'Suburban' # 3203 households / 7.14 sq. mi = 449 households / sq mi (Suburban)

# %%
killings.loc[3581, 'geo_type'] = 'Suburban' # 125894 households / 740 sq. mi = 170 households / sq mi (Suburban)

# %% [markdown]
# The next location is at Foxwoods Casino in CT, which is technically part of Mashantucket CT., I think because it's considered an indian reservation.  The casino is surrounded by Ledyard CT. and it might be a better idea to use that city as it may be more representative of the household popluation size and square mileage need to classify the geo_type.

# %%
killings.loc[3621, 'geo_type'] = 'Rural' # 62 households / 2.6 sq. mi = 24 households / sq mi (Rural)

# %%
killings.loc[3699, 'geo_type'] = 'Rural' # 115 households / 1.5 sq. mi = 77 households / sq mi (Rural)

# %%
killings.loc[3740, 'geo_type'] = 'Suburban' # 521198 households / 385.8 sq. mi = 1351 households / sq mi (Suburban)

# %%
killings.loc[4409, ['geo_type', 'zipcode']] = 'Suburban', 32218.0 #359607 households / 875 sq mi = 411 (Suburban)

# %%
killings.loc[4535, ['geo_type', 'zipcode']] = 'Suburban', 46368.0 #13992 households / 27.61 sq mi = 507 (Suburban)

# %%
killings.loc[4571, ['geo_type', 'zipcode']] = 'Suburban', 97210.0 #264428 households / 145 sq mi = 1824 (Suburban)

# %%
killings.loc[[4592, 4593], ['geo_type', 'zipcode']] = 'Suburban', 77014.0 #848340 households / 669 sq mi = 1268 (Suburban)

# %%
killings.loc[4594, ['geo_type', 'zipcode']] = 'Suburban', 74434.0 #1639 households / 14.13 sq mi = 116 (Suburban)

# %%
killings.loc[4640, ['geo_type', 'zipcode']] = 'Suburban', 30680.0 #5337 households / 14.15 sq mi = 377 (Suburban)

# %%
killings.loc[5021, 'geo_type'] = 'Rural' # 268 households / 8.842 sq. mi = 30 households / sq mi (Rural)

# %%
killings.loc[5164, ['geo_type', 'street_address']] = 'Rural', '182 N 4430 Rd' # No census data on household population, but on Google maps it looks very rural

# %%
killings.loc[5192, 'geo_type'] = 'Suburban' # 39122 households / 22.99 sq. mi = 1702 households / sq mi (Suburban)

# %%
killings.loc[5268, 'geo_type'] = 'Suburban' # 355 households / 0.74 sq. mi = 480 households / sq mi (Suburban)

# %%
killings.loc[5371, ['geo_type', 'zipcode']] = 'Suburban', 77073.0 #848340 households / 669 sq mi = 1268 (Suburban)

# %%
killings.loc[5623, 'geo_type'] = 'Suburban' # 63217 households / 103.1 sq. mi = 613 households / sq mi (Suburban)

# %%
killings.loc[5709, 'geo_type'] = 'Rural' # Jean is just outside of Las Vegas.  Has no residents but is considered a commercial town.  Seems rural enough.

# %%
killings.loc[5805, 'geo_type'] = 'Suburban' # 3061 households / 13.13 sq. mi = 233 households / sq mi (Suburban)

# %%
killings.loc[[4451, 6080], 'geo_type'] = 'Urban' # 323446 households / 142.5 sq. mi = 2270 households / sq mi (Urban)

# %%
killings.loc[6188, 'geo_type'] = 'Urban' # 7229 households / 0.648 sq. mi = 11156 households / sq mi (Urban)\

# %%
killings.loc[6570, ['street_address', 'zipcode', 'geo_type']] = '12097 Veterans Memorial Dr', 77067.0, 'Suburban' #848340 households / 669 sq mi = 1268 (Suburban)

# %%
killings.loc[6442, 'geo_type'] = 'Rural' # Outskirts of Las Vegas, seems very Rural

# %%
killings.loc[6573, ['geo_type', 'zipcode', 'city']] = 'Suburban', 73104.0, 'Oklahoma City' # 240471 / 621 = 387 (Suburban)

# %%
killings.loc[6637, ['geo_type', 'zipcode']] = 'Suburban', 70767.0 # 2162 / 3.328 = 650 (Suburban)

# %%
killings.loc[6643, ['street_address', 'zipcode', 'geo_type']] = '32000 Westport Way', 92596.0, 'Suburban' # 8539 / 10.9 = 783 (Suburban)

# %%
killings.loc[6697, ['street_address', 'geo_type']] = '2335 Union Dr', 'Suburban' # 25243 / 224.27 = 113 (Suburban)

# %%
killings.loc[6746, 'geo_type'] = 'Suburban' # 199478 / 136.8 = 1458 (Suburban)

# %%
killings.loc[6848, 'geo_type'] = 'Urban' # 281322 / 68.34 = 4117 (Urban)

# %%
killings.loc[6862, ['zipcode', 'geo_type']] = 15224.0, 'Urban' # 136275 / 58.34 = 2336

# %%
killings.loc[6933, 'geo_type'] = 'Suburban' # 113901 / 108 = 1055 (Suburban)

# %%
killings.loc[killings['geo_type'].isnull()==True, ['street_address', 'city', 'state', 'zipcode', 'geo_type', 'news_article_link']]

# %%
killings.loc[1029]

# %%
killings.loc[528, 'geo_type'] = 'Suburban' # 53925 / 128.4 = 420 (Suburban)

# %%
killings.loc[774, 'geo_type'] = 'Suburban' # 40035 / 34.11 = 1174 (Suburban)

# %%
# killings.loc[1029, 'geo_type'] = 'Suburban' # 682 / 3.14 = 217 (Suburban)

# %%
killings.loc[1250, ['street_address', 'zipcode', 'geo_type']] = '500 N Main St', 67124.0, 'Suburban' # 2837 / 7.49 = 379 (Suburban)

# %%
killings.loc[1305, 'geo_type'] = 'Suburban' # 2810 / 11.54 = 244 (Suburban)

# %%
killings.loc[1322, 'geo_type'] = 'Suburban' # 574 / 1.324 = 434 (Suburban)

# %%
killings.loc[1336, 'geo_type'] = 'Suburban' # 28177 / 18.57 = 1517 (Suburban)

# %%
killings.loc[1346, 'geo_type'] = 'Urban' # 229556 / 96.81 = 2371 (Urban)

# %%
killings.loc[1356, 'geo_type'] = 'Suburban' # 1258 / 2.31 = 545 (Suburban)

# %%
killings.loc[1367, 'geo_type'] = 'Rural' # 394 / 8.452 = 47 (Rural)

# %%
killings.loc[1430, 'geo_type'] = 'Suburban' # 10780 / 8.687 = 1241 (Suburban)

# %%
killings.loc[1607, 'geo_type'] = 'Suburban' # 23121 / 19.7 = 1174 (Suburban)

# %%
killings.loc[1812, ['geo_type']] = 'Rural' # 32 / .4 = 80 (Rural)

# %%
killings.loc[1965, 'geo_type'] = 'Suburban' # 111221 / 184.4 = 603 (Suburban)

# %%
killings.loc[1981, 'geo_type'] = 'Suburban' # 118 / 1.05 = 112 (Suburban)

# %%
killings.loc[2813, ['street_address', 'zipcode', 'geo_type']] = '6800 62nd Ave NE', 98115.0, 'Urban' # 283510 / 83.78 = 3384 (Urban)

# %%
killings.loc[3344, ['city', 'zipcode', 'geo_type']] = 'Campbellton', 78008.0, 'Rural' # 176 / 198.5 = 1 (Rural)

# %%
killings.loc[3346, ['zipcode', 'geo_type']] = 57752.0, 'Rural' # 177 / 2.008 = 88 (Rural)

# %%
killings.loc[3475, ['street_address', 'zipcode', 'geo_type', 'city']] = 'X4 Rd', 81411.0, 'Rural', 'Bedrock' #Out in the middle of nowhere

# %%
killings.loc[6812, 'geo_type'] = 'Suburban' # 4034 / 17.1 = 236 (Suburban)

# %%
killings.loc[7099, 'geo_type'] = 'Urban' # 870051 / 55.25 = 15748 (Urban)

# %%
killings.loc[7461, 'geo_type'] = 'Suburban' # 8689 / 4.36 = 1993 (Suburban)

# %% [markdown]
# After going through all the links, we'll replace all null values that remain with None

# %%
num_null_street_address = killings['street_address'].isnull().sum()
print(F"There are {num_null_street_address} street addresses that will be filled with 'None'")
killings['street_address'].fillna('None', inplace=True)

# %%
killings.isnull().sum()

# %%
killings.loc[6933, 'news_article_link']

# %%
killings.loc[6933]

# %%
killings.loc[7461, 'news_article_link']

# %%
killings.loc[7461]

# %%

# %%
killings.isnull().sum()

# %%
killings.loc[killings['official_disposition_of_death'].isnull()==True, 
             'official_disposition_of_death'] = 'Unknown'

# %%
killings.isnull().sum()

# %%
killings.loc[killings['Agency responsible for death'].isnull()==True, 'Agency responsible for death'] = 'Unknown'

# %%
killings.isnull().sum()

# %% [markdown]
# Since we've done as much as we can in the way of researching population densities and manually entering the geo_type.  We'll impute the remaing missing values (only 1 entry) with the mode of that column (Suburban)

# %%
killings['geo_type'].value_counts()

# %%
#killings.loc[killings['geo_type'].isnull() == True, 'geo_type']
mode = killings['geo_type'].mode()[0]
# killings.loc[killings['geo_type'].isnull() == True, 'geo_type'] = str(mode)
# killings.loc[killings['geo_type'].isnull() == True, 'geo_type']
killings['geo_type'].fillna(mode, inplace=True)

# %%
killings.isnull().sum()

# %%
killings.loc[1755]

# %% [markdown]
# Since row 1755 has so many missing values, contains so little information, and is the one entry remaining with NaN for the city, I'm going to drop it.

# %%
killings.drop(labels=1755, inplace=True)

# %%
killings.isnull().sum()

# %%
killings.loc[killings["victims_age"]=='Unknown', "victims_age"] = np.nan

# %%
killings['victims_age'].unique()

# %%
killings.loc[killings["victims_age"]=='40s', "victims_age"] = 40.0

# %%
killings.dtypes

# %%
killings["victims_age"] = killings["victims_age"].astype('float64')

# %%
killings.hist("victims_age")
plt.show()

# %%
median_age = round(killings["victims_age"].median())
killings["victims_age"] = killings["victims_age"].fillna(round(median_age))

# %%
killings.isnull().sum()

# %%
killings.info()

# %%
zipcode_null = killings.loc[killings['Zipcode'].isnull()==True,:].index
killings.drop(zipcode_null,inplace=True)

# %%
killings.isnull().sum()

# %%
killings['official_disposition_of_death'].value_counts()

# %%
killings['official_disposition_of_death'].unique()

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('unjustified'), 'official_disposition_of_death'] = 'Unjustified'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('justified'), 'official_disposition_of_death'] = 'Justified'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('convicted'), 'official_disposition_of_death'] = 'Convicted'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('acquitted'), 'official_disposition_of_death'] = 'Acquitted'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('charged'), 'official_disposition_of_death'] = 'Charged'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('pending investigation'), 'official_disposition_of_death'] = 'Pending Investigation'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('unknown'), 'official_disposition_of_death'] = 'Unknown'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('pending investigaton'), 'official_disposition_of_death'] = 'Pending Investigation'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('ongoing investigation'), 'official_disposition_of_death'] = 'Under Investigation'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('unknown'), 'official_disposition_of_death'] = 'Unknown'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('no indictment'), 'official_disposition_of_death'] = 'No indictment'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('No charges'), 'official_disposition_of_death'] = 'No charges'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('unreported'), 'official_disposition_of_death'] = 'Unreported'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('under investigation'), 'official_disposition_of_death'] = 'Under Investigation'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('indicted'), 'official_disposition_of_death'] = 'Indicted'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('no charges'), 'official_disposition_of_death'] = 'No charges'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('no known charges'), 'official_disposition_of_death'] = 'No charges'

# %%
killings['official_disposition_of_death'].value_counts()

# %%
killings.head()

# %%
killings['criminal_charges'].value_counts()

# %%
killings.loc[killings['criminal_charges'] == 'No', 'criminal_charges'] = 'No Charges'
killings.loc[killings['criminal_charges'] == 'NO', 'criminal_charges'] = 'No Charges'
killings.loc[killings['criminal_charges'] == 'No known charges', 'criminal_charges'] = 'No Charges'

# %%
killings['criminal_charges'].value_counts()

# %%
convicted = killings[killings['criminal_charges'].str.lower().str.contains('convicted')]
for i, index in enumerate(convicted.index):
    print()
    print(str(i+1) + '. ' + killings.iloc[index]['desc_of_circumstances'])
    print(killings.iloc[index]['news_article_link'])
    print()

# %%
killings.loc[killings["victims_race"] == 'Unknown race', "victims_race"] = 'Unknown'
killings.loc[killings["victims_race"] == 'Unknown Race', "victims_race"] = 'Unknown'

# %%
killings.loc[killings["victims_race"]=='Asian', "victims_race"] = 'Asian/Pacific Islander'
killings.loc[killings["victims_race"]=='Pacific Islander', "victims_race"] = 'Asian/Pacific Islander'

# %%
killings["victims_race"].value_counts()

# %%
killings['victims_name'].value_counts()

# %%
killings[['First Name', 'Last Name']] = killings['victims_name'].loc[killings['victims_name'].str.split().str.len() == 2].str.split(expand=True)

killings.loc[killings['victims_name'].str.split().str.len() != 2, 'First Name'] = killings['victims_name'].str.split().str[0]
killings.loc[killings['victims_name'].str.split().str.len() != 2, 'Last Name'] = killings['victims_name'].apply(lambda x: HumanName(x).last)
# killings.loc[killings['victims_name'].str.split().str.len() != 2, ['First Name', 'Last Name']] = [killings['victims_name'].str.split().str[0], killings['victims_name'].str.split().str[-1]]

killings.loc[killings['victims_name'] == 'Name withheld by police', ['First Name', 'Last Name']] = ['Unknown', 'Unknown']

killings[['First Name', 'Last Name']]

# %%
killings['First Name'].value_counts()[0:50]

# %%
killings['Last Name'].value_counts()[0:60]

# %%
killings.head()

# %%
killings.to_csv('./csv_files/police_killings_clean.csv', index=False)

# %%

# %%

# %% [markdown]
# # Scratch Work

# %%
# This code will open all links so we can inspect them manually
try:
    for url in null_gender_urls:
        webbrowser.open_new_tab(url)
except:
    pass

# %%
import urllib.request

def is_url_working(x):
    import urllib.request
    print('starting')
    try:
        test_url = urllib.request.urlopen(x)
        return test_url
    except:
        return 'No'
    
# killings['URL working?'] = killings['Link to news article or photo of official document'].apply(is_url_working)


# %%
null_address_df = killings[(killings["street_address"].isnull() == True) & (killings["news_article_link"].isnull()==False)]

null_address_links = null_address_df['news_article_link']

null_address_idx = null_address_links.index
null_address_urls = null_address_links.to_list()

print("Here are the links for news articles that contain links for rows with a null street_address:")
print("-"*50)
print()
for idx, link in zip(null_address_idx[0:10], null_address_links[0:10]):
    null_col_mask = killings.loc[idx].isna()==True
    print(F"df_idx: {idx}")
    print(killings.loc[idx, null_col_mask].to_string())
    print(F"link: {link}")
    print()
