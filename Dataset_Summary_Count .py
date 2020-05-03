#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd

current_directory = os.getcwd()
dataset_directory = os.path.join(current_directory,'Dataset_skin')
if not os.path.exists(dataset_directory):
	print('Error : No Dataset Found')

label_name = []
count = []

for filename in os.listdir(dataset_directory):
	cwd = os.path.join(dataset_directory,filename)
	check_dir = os.path.join(cwd,'Original')
	label_name.append(filename)
	count.append(len(os.listdir(check_dir)))

counts = pd.DataFrame({'Label':label_name,'Count': count})
print(counts)
print('Total Data Images: ',sum(count))


# In[ ]:




