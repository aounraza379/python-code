import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron 
from sklearn.metrics import accuracy_score

data = pd.read_csv('placement_dataset.csv')
print(data.head(3))