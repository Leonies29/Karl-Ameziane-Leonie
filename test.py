import numpy as np # linear algebra
import pandas as pd 
from sklearn.datasets import fetch_20newsgroups
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

newspaper = fetch_20newsgroups(subset='all',remove=('headers','footers','qutoes'),return_X_y=False)

pd = pd.DataFrame({
    'text':newspaper.data,
    'target':newspaper.target,
    'category':[newspaper.target_names[i] for i in newspaper.target]})