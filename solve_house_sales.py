"""
Populate solutions into House_Sales_in_King_Count_USA.ipynb without modifying comments or data.
"""
import json

with open('House_Sales_in_King_Count_USA.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 10: safe piplite import
nb['cells'][10]['source'] = [
    "try:\n",
    "    import piplite\n",
    "    await piplite.install('seaborn')\n",
    "except Exception:\n",
    "    pass\n",
    "\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler,PolynomialFeatures\n",
    "from sklearn.linear_model import LinearRegression\n",
    "%matplotlib inline\n"
]

# Cell 13: safe download definition
nb['cells'][13]['source'] = [
    "import os\n",
    "try:\n",
    "    from pyodide.http import pyfetch\n",
    "    async def download(url, filename):\n",
    "        response = await pyfetch(url)\n",
    "        if response.status == 200:\n",
    "            with open(filename, \"wb\") as f:\n",
    "                f.write(await response.bytes())\n",
    "except Exception:\n",
    "    import urllib.request\n",
    "    async def download(url, filename):\n",
    "        if not os.path.exists(filename):\n",
    "            urllib.request.urlretrieve(url, filename)\n"
]

# Cell 15: safe await download
nb['cells'][15]['source'] = [
    "if not os.path.exists(\"housing.csv\"):\n",
    "    await download(filepath, \"housing.csv\")\n",
    "file_name=\"housing.csv\"\n"
]

# Cell 23 (Question 1)
nb['cells'][23]['source'] = [
    "#Enter Your Code, Execute and take the Screenshot\n",
    "df.dtypes\n"
]

# Cell 28 (Question 2)
nb['cells'][28]['source'] = [
    "#Enter Your Code, Execute and take the Screenshot\n",
    "df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)\n",
    "df.describe()\n"
]

# Cell 32 (Replace bedrooms NaNs safely for pandas 3.0)
nb['cells'][32]['source'] = [
    "mean = df['bedrooms'].mean()\n",
    "df['bedrooms'] = df['bedrooms'].replace(np.nan, mean)\n"
]

# Cell 34 (Replace bathrooms NaNs safely for pandas 3.0)
nb['cells'][34]['source'] = [
    "mean = df['bathrooms'].mean()\n",
    "df['bathrooms'] = df['bathrooms'].replace(np.nan, mean)\n"
]

# Cell 38 (Question 3)
nb['cells'][38]['source'] = [
    "#Enter Your Code, Execute and take the Screenshot\n",
    "df['floors'].value_counts().to_frame()\n"
]

# Cell 40 (Question 4)
nb['cells'][40]['source'] = [
    "sns.boxplot(x='waterfront', y='price', data=df)\n"
]

# Cell 42 (Question 5)
nb['cells'][42]['source'] = [
    "#Enter Your Code, Execute and take the Screenshot\n",
    "sns.regplot(x='sqft_above', y='price', data=df)\n"
]

# Cell 49 (Question 6)
nb['cells'][49]['source'] = [
    "#Enter Your Code, Execute and take the Screenshot\n",
    "X = df[['sqft_living']]\n",
    "Y = df['price']\n",
    "lm = LinearRegression()\n",
    "lm.fit(X, Y)\n",
    "lm.score(X, Y)\n"
]

# Cell 53 (Question 7)
nb['cells'][53]['source'] = [
    "#Enter Your Code, Execute and take the Screenshot\n",
    "lm.fit(df[features], df['price'])\n",
    "lm.score(df[features], df['price'])\n"
]

# Cell 57 (Question 8)
nb['cells'][57]['source'] = [
    "#Enter Your Code, Execute and take the Screenshot\n",
    "pipe = Pipeline(Input)\n",
    "pipe.fit(df[features], df['price'])\n",
    "pipe.score(df[features], df['price'])\n"
]

# Cell 65 (Question 9)
nb['cells'][65]['source'] = [
    "#Enter Your Code, Execute and take the Screenshot\n",
    "RidgeModel = Ridge(alpha=0.1)\n",
    "RidgeModel.fit(x_train, y_train)\n",
    "RidgeModel.score(x_test, y_test)\n"
]

# Cell 67 (Question 10)
nb['cells'][67]['source'] = [
    "#Enter Your Code, Execute and take the Screenshot\n",
    "pr = PolynomialFeatures(degree=2)\n",
    "x_train_pr = pr.fit_transform(x_train)\n",
    "x_test_pr = pr.transform(x_test)\n",
    "RidgeModel = Ridge(alpha=0.1)\n",
    "RidgeModel.fit(x_train_pr, y_train)\n",
    "RidgeModel.score(x_test_pr, y_test)\n"
]

with open('House_Sales_in_King_Count_USA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Populated all questions into House_Sales_in_King_Count_USA.ipynb!")
