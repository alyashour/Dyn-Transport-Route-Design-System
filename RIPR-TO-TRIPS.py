#!/usr/bin/env python
# coding: utf-8

# In[34]:


riders_matrix, tpm, shape_matrix, total_riders = ripr_output(2021, 11, 18, "clear")


# In[35]:


import pandas as pd
stops = pd.read_csv("stops.txt")
stop_ids = stops["stop_id"].tolist()


# In[36]:


import numpy as np

def show_top_flows(riders_matrix, stop_ids, top_k=30):
    n = riders_matrix.shape[0]
    flows = []

    for i in range(n):
        for j in range(n):
            riders = riders_matrix[i, j]
            if riders > 0:  # any positive expected riders
                flows.append((i, j, riders))

    # Sort by riders descending
    flows.sort(key=lambda x: x[2], reverse=True)

    print(f"Found {len(flows)} OD pairs with > 0 expected riders")
    print(f"Top {top_k} flows:\n")
    for i, j, r in flows[:top_k]:
        print(f"{stop_ids[i]} → {stop_ids[j]}: {r:.3f} expected riders")

show_top_flows(riders_matrix, stop_ids, top_k=30)


# In[37]:


rng = np.random.default_rng(0)  # fixed seed for reproducibility
sampled_trips = rng.poisson(riders_matrix)  # same shape, but integers

def show_top_flows_sampled(sampled_matrix, stop_ids, top_k=30):
    n = sampled_matrix.shape[0]
    flows = []

    for i in range(n):
        for j in range(n):
            riders = sampled_matrix[i, j]
            if riders > 0:  # at least 1 rider simulated
                flows.append((i, j, riders))

    flows.sort(key=lambda x: x[2], reverse=True)

    print(f"Found {len(flows)} OD pairs with ≥ 1 simulated rider")
    print(f"Top {top_k} flows:\n")
    for i, j, r in flows[:top_k]:
        print(f"{stop_ids[i]} → {stop_ids[j]}: {r} riders")

show_top_flows_sampled(sampled_trips, stop_ids, top_k=30)


# In[38]:


print("total_riders:", total_riders)
print("riders_matrix sum:", riders_matrix.sum(), "max:", riders_matrix.max())


# In[ ]:




