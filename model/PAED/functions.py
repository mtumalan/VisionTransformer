import numpy as np
import pandas as pd

#def load_classdict(path):
#    classdict = pd.read_csv(path)
#    rgb_to_class = {
#        (row['r'], row['g'], row['b']): idx
#        for idx, row in classdict.iterrows()
#    }
#    return rgb_to_class

def load_classdict(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, sep='\t') if '\t' in open(csv_path).readline() else pd.read_csv(csv_path)
    class_dict = {}
    class_names = []
    for idx, row in enumerate(df.itertuples(index=False)):
        rgb = (row.r, row.g, row.b)
        class_dict[rgb] = idx
        class_names.append(row.name)
    return class_dict, class_names

def convertBW(rgb_to_class):
    bw_dict = {}
    for rgb, cls in rgb_to_class.items():
        bw_dict[cls] = np.mean(rgb)

    return bw_dict

def assign_closest_class(value, bw_dict):
    closest_class = None
    min_difference = float('inf')

    for cls, bw_value in bw_dict.items():
        difference = abs(value - bw_value)
        if difference < min_difference:
            min_difference = difference
            closest_class = cls

    return closest_class
