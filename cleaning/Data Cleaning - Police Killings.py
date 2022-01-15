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
    'Body Camera (Source: WaPo)':'video_surveillance',
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

# %% [markdown] tags=[]
# ## Cleaning column by column

# %% [markdown]
# ### Victim's name - splitting to first and last names

# %%
names_without_police_in_them = killings.loc[killings['victims_name'].str.contains('police')==False, 'victims_name']

killings['first_name'] = names_without_police_in_them.apply(lambda x: HumanName(x).first)

killings['last_name'] = names_without_police_in_them.apply(lambda x: HumanName(x).last)

# %%
# killings[['First Name', 'Last Name']] = killings['victims_name'].loc[killings['victims_name'].str.split().str.len() == 2].str.split(expand=True)

# killings.loc[killings['victims_name'].str.split().str.len() != 2, 'First Name'] = killings['victims_name'].str.split().str[0]
# killings.loc[killings['victims_name'].str.split().str.len() != 2, 'Last Name'] = killings['victims_name'].apply(lambda x: HumanName(x).last)

# killings.loc[killings['victims_name'] == 'Name withheld by police', ['First Name', 'Last Name']] = ['Unknown', 'Unknown']

# killings[['First Name', 'Last Name']]

# %% [markdown]
# ### Victim's age

# %%
killings['victims_age'].unique()

# %%
killings.loc[killings["victims_age"]=='Unknown', "victims_age"] = np.nan

# %%
killings.loc[killings["victims_age"]=='40s', "victims_age"] = 40.0

# %% [markdown]
# ### Victim's gender
# Let's try to determine what to do with the Null values for gender.  We'll begin by inspecting those that have a news article link present as that may have information we can use.

# %%
killings['victims_gender'].unique()

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
killings.loc[13, 'victims_gender'] = 'male'

# %%
# news article says the victim was male in his 40s
killings.loc[112, ['victims_gender', 'victims_age']] = 'male', '40'

# %%
# URL mentions male victim, but ad is behind paywall so can't investigate further
killings.loc[1029, 'victims_gender'] = 'male'

# %%
killings.loc[774]

# %%
killings[(killings['victims_gender'].isnull()==True) & (killings['news_article_link'].isnull()==True)]

# %%
# Both names sound male, so I'll set the gender accordingly
killings.loc[528, 'victims_gender'] = 'male'
killings.loc[774, 'victims_gender'] = 'male'

# %%
num_null_gender = killings['victims_gender'].isnull().sum()
print(F"There are only {num_null_gender} rows with a null gender left, so we'll impute 'Unknown'")

killings['victims_gender'].fillna('Unknown', inplace=True)

# %%
killings['victims_gender'] = killings['victims_gender'].str.lower()

# %% [markdown]
# ### Victim's race

# %%
killings['victims_race'].unique()

# %%
killings['victims_race'] = killings['victims_race'].str.lower()

# %%
killings.loc[killings["victims_race"] == 'unknown race', "victims_race"] = 'unknown'

# %%
killings.loc[killings["victims_race"]=='asian', "victims_race"] = 'asian/pacific islander'
killings.loc[killings["victims_race"]=='pacific islander', "victims_race"] = 'asian/pacific islander'

# %% [markdown]
# ### Victim Image URL

# %% [markdown]
# For rows that don't have a URL image of the victim, I'm going to impute "None"

# %%
killings['victim_img_url'].fillna('None', inplace=True)

# %% [markdown]
# ### Location Data

# %% [markdown]
# #### City

# %%
killings.loc[killings['city'].isnull()==True,['city', 'state', 'zipcode']]

# %% [markdown]
# Since we have a zipcode and state for several of these rows, we can look up the appropriate city

# %%
killings.loc[3339, 'city'] = "Land O' Lakes"
killings.loc[5561, 'city'] = "Jacksonville"
killings.loc[6511, 'city'] = "Douglas"

# %% [markdown]
# #### County

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
# #### Geography Type/Zipcode/Street Address

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
killings.loc[528, 'geo_type'] = 'Suburban' # 53925 / 128.4 = 420 (Suburban)

# %%
killings.loc[774, 'geo_type'] = 'Suburban' # 40035 / 34.11 = 1174 (Suburban)

# %%
killings.loc[1029, 'geo_type'] = 'Suburban' # 682 / 3.14 = 217 (Suburban)

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
# #### Handling location fields that are still Null
# Since we've done as much as we can in the way of researching location data, we'll replace all null values that remain with 'unknown'

# %%
num_null_street_address = killings['street_address'].isnull().sum()
print(F"There are {num_null_street_address} null street addresses that will be filled with 'unknown'")
killings['street_address'].fillna('unknown', inplace=True)

# %%
killings.loc[killings['street_address']=='Unknown', 'street_address'] = 'unknown'

# %%
num_null_city = killings['city'].isnull().sum()
print(F"There are {num_null_city} cities that will be filled with 'unknown'")

killings['city'].fillna('unknown', inplace=True)

# %%
# num_null_zip_codes = killings['zipcode'].isnull().sum()
# print(F"There are {num_null_zip_codes} zip_codes that will be filled with 'unknown'")

# killings['zipcode'].fillna('unknown', inplace=True)

# %%
num_null_geo_type = killings['geo_type'].isnull().sum()
print(F"There are {num_null_geo_type} geo_types that will be filled with 'unknown'")

killings['geo_type'].fillna('unknown', inplace=True)

# %% [markdown]
# ### Agency Responsible for death

# %%
num_null_agency_responsible = killings['agency_resp_for_death'].isnull().sum()
print(F"There are {num_null_agency_responsible} rows for agency responsible for death that will be filled with 'unknown'")

killings['agency_resp_for_death'].fillna('unknown', inplace=True)

# %% [markdown]
# ### Cause of death
# There are too many categories here to do a meaningful analysis, I'll distill them down to just a few categories

# %%
killings['cause_of_death'].unique()

# %%
killings['cause_of_death'] = killings['cause_of_death'].str.lower()

# %%
cause_of_death_dict = {"gunshot, bean bag gun":"gunshot, beanbag gun",
                       "tasered":"taser",
                       "beaten/bludgeoned with instrument":"beaten",
                       "gunshot, taser":"gunshot",
                       "gunshot, police dog":"gunshot",
                       "gunshot, pepper spray":"gunshot",
                       "gunshot, beanbag gun":"gunshot",
                       "taser, pepper spray, beaten":"taser",
                       "taser, physical restraint":"taser",
                       "gunshot, taser, pepper spray":"gunshot",
                       "gunshot, stabbed":"gunshot",
                       "gunshot, vehicle":"gunshot",
                       "gunshot, taser, baton":"gunshot",
                       "gunshot, unspecified less lethal weapon":"gunshot",
                       "gunshot, taser, beanbag shotgun":"gunshot",
                       "gunshot, taser, beanbag shotgun":"gunshot",
                       "taser, baton":"taser",
                       "taser, beaten":"taser",
                       "bomb":"other",
                       "baton, pepper spray, physical restraint":"other",
                       "pepper spray":"other",
                       "bean bag":"other"}

killings['cause_of_death'] = killings['cause_of_death'].map(cause_of_death_dict).fillna(killings['cause_of_death'])

# %%
killings['cause_of_death'].value_counts()

# %% [markdown]
# ### Description of Circumstances Surrounding Death
# It'll take too much time to research these missing entries and I don't think they'll contribute much to analysis, so I'll fill them with "Unavailable"

# %%
killings['desc_of_circumstances'].fillna('Unavailable', inplace=True)

# %% [markdown]
# ### Official Disposition of Death

# %%
killings['official_disposition_of_death'].unique()

# %% [markdown]
# There are way too many unique entries in this column, let's reduce the number of categories down

# %%
# off_disp_dict = {"pending investigaton":"pending investigation",
#                  "ongoing investigation":"under investigation",
#                  }

# %%
num_null_disp_of_death = killings['official_disposition_of_death'].isnull().sum()
print(F"There are {num_null_disp_of_death} rows for official disposition of death that will be filled with 'unknown'")

killings['official_disposition_of_death'].fillna('unknown', inplace=True)

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('unjustified'), 'official_disposition_of_death'] = 'unjustified'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('justified'), 'official_disposition_of_death'] = 'justified'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('convicted'), 'official_disposition_of_death'] = 'convicted'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('acquitted'), 'official_disposition_of_death'] = 'acquitted'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('pending investigation'), 'official_disposition_of_death'] = 'pending investigation'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('pending investigaton'), 'official_disposition_of_death'] = 'pending investigation'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('ongoing investigation'), 'official_disposition_of_death'] = 'under investigation'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('under investigation'), 'official_disposition_of_death'] = 'under investigation'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('no indictment'), 'official_disposition_of_death'] = 'no indictment'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('indicted'), 'official_disposition_of_death'] = 'indicted'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('charged'), 'official_disposition_of_death'] = 'charged'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('No charges'), 'official_disposition_of_death'] = 'no charges'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('no charges'), 'official_disposition_of_death'] = 'no charges'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('no known charges'), 'official_disposition_of_death'] = 'no charges'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('unreported'), 'official_disposition_of_death'] = 'unreported'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('unknown'), 'official_disposition_of_death'] = 'unknown'

# %%
killings.loc[killings['official_disposition_of_death'].str.lower().str.contains('unknown'), 'official_disposition_of_death'] = 'unknown'

# %%
killings['official_disposition_of_death'] = killings['official_disposition_of_death'].str.lower()

# %%
killings['official_disposition_of_death'].value_counts()

# %% [markdown]
# ### Criminal Charges

# %%
killings['criminal_charges'].value_counts()

# %%
killings['criminal_charges'] = killings['criminal_charges'].str.lower()

# %%
killings.loc[killings['criminal_charges'] == 'no known charges', 'criminal_charges'] = 'no charges'
killings.loc[killings['criminal_charges'] == 'no', 'criminal_charges'] = 'no charges'

# %%
killings.loc[killings['criminal_charges'].str.contains('charged, convicted'), 'criminal_charges'] = 'charged, convicted'

# %%
killings.loc[killings['criminal_charges'].str.contains('charged, mistrial'), 'criminal_charges'] = 'charged, mistrial'

# %%
killings.loc[killings['criminal_charges'].str.contains('charged, charges tossed'), 'criminal_charges'] = 'charged, charges dropped'

# %%
killings.loc[killings['criminal_charges'].str.contains('charged with manslaughter'), 'criminal_charges'] = 'charged with a crime'

# %%
killings['criminal_charges'].value_counts()

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
killings['mental_illness'] = killings['mental_illness'].str.lower()

# %%
mental_illness_dict = {"unkown":"unknown",
                       "unknown ":"unknown"}

killings['mental_illness'] = killings['mental_illness'].map(mental_illness_dict).fillna(killings['mental_illness'])

# %%
num_null_killings = killings['mental_illness'].isnull().sum()
print(F"There are {num_null_killings} null rows for mental_illness that will be filled with 'unknown'")

killings['mental_illness'].fillna('unknown', inplace=True)

# %% [markdown]
# ### Unarmed

# %%
killings['unarmed'].unique()

# %%
killings['unarmed'] = killings['unarmed'].str.lower()

# %% [markdown]
# ### Alleged Weapon - In Progress
# I'll distill down some of the repeats here, but I may want to come back to this later and distill things further down.  Some ideas:
# - [ ] Gun category
# - [ ] Tool category
# - [ ] Baseball bat category
# - [ ] Knife category
# - [ ] Toy category
# - [ ] Metal object category
# - [ ] Sharp object category

# %%
killings['alleged_weapon'].sort_values().unique()

# %%
killings['alleged_weapon'] = killings['alleged_weapon'].str.lower()

# %%
killings['alleged_weapon'] = killings['alleged_weapon'].str.rstrip()

# %%
alleged_weapon_dict = {"unclear":"unknown",
                       "unknown weapon":"unknown",
                       "undetermined":"unknown",
                       "unknown object":"unknown",
                       "air pistol":"airsoft pistol",
                       "knife and gun":"gun and knife",
                       "ax":"axe",
                       "chain saw":"chainsaw",
                       "flag pole":"flagpole",
                       "gun and knives":"gun and knife",
                       "gun and car":"gun and vehicle",
                       "gun, vehicle":"gun and vehicle",
                       "guns":"gun",
                       "knives":"knife",
                       "rocks":"rock",
                       "screw driver":"screwdriver",
                       "sticks":"stick",
                       "unclear weapon":"unknown",
                       "wood stick":"wooden stick",
                       "bat":"baseball bat",
                       "blunt weapon":"blunt object",
                       "hammer and knife":"knife and hammer",
                       "knife/scissors":"knife and scissors"}

killings['alleged_weapon'] = killings['alleged_weapon'].map(alleged_weapon_dict).fillna(killings['alleged_weapon'])

# %% [markdown]
# ### Threat Level

# %%
killings['threat_level'].unique()

# %%
num_null_threat_level = killings['threat_level'].isnull().sum()
print(F"There are {num_null_threat_level} rows for threat level that will be filled with 'unknown'")

killings['threat_level'].fillna('unknown', inplace=True)

# %% [markdown]
# ### Fleeing

# %%
killings['fleeing'].unique()

# %%
killings['fleeing'] = killings['fleeing'].str.lower()

killings.loc[killings['fleeing']=='0', 'fleeing'] = 'unknown'
# fleeing_dict = {'Foot':'foot',
#                 'Not fleeing':'not fleeing',
#                 '0':'unknown',
#                 'Car':'car',
#                 'Other':'other'}

# killings['fleeing'] = killings['fleeing'].map(fleeing_dict).fillna(killings['fleeing'])
killings['fleeing'].fillna('unknown', inplace=True)

# %% [markdown]
# ### Video surveillance

# %%
killings['video_surveillance'].unique()

# %%
killings['video_surveillance'] = killings['video_surveillance'].str.lower()
killings['video_surveillance'] = killings['video_surveillance'].str.replace('yes', 'body camera')
killings['video_surveillance'].fillna('unknown', inplace=True)

# %% [markdown]
# ## Converting data types

# %%
killings.dtypes

# %% [markdown]
# ### Age

# %%
killings["victims_age"] = killings["victims_age"].astype('float64')

# %% [markdown]
# ### Date

# %% [markdown]
# We can convert the dates in the date column to Python datetime objects

# %%
killings["date"] = pd.to_datetime(killings["date"], infer_datetime_format=True)

# %% [markdown]
# # Saving work to CSV file

# %%
killings.to_csv('./csv_files/police_killings_clean.csv', index=False)

# %% [markdown]
# # Scratch Work

# %%
# null_address_df = killings[(killings["street_address"].isnull() == True) & (killings["news_article_link"].isnull()==False)]

# null_address_links = null_address_df['news_article_link']

# null_address_idx = null_address_links.index
# null_address_urls = null_address_links.to_list()

# print("Here are the links for news articles that contain links for rows with a null street_address:")
# print("-"*50)
# print()
# for idx, link in zip(null_address_idx[0:10], null_address_links[0:10]):
#     null_col_mask = killings.loc[idx].isna()==True
#     print(F"df_idx: {idx}")
#     print(killings.loc[idx, null_col_mask].to_string())
#     print(F"link: {link}")
#     print()
