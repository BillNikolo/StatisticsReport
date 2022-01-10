import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score

case_study = {1: "(A2.a) Descriptive Statistics",
              2: "(A2.b) Scatterplot: “Duration_window_flow” x “Avg_pkts_lenght”",
              3: "(A2.c) Boxplot: “Avg_pkts_lenght”",
              4: "(A2.d(1)) Pearson Coefficient 'Duration_window_flow' x 'Avg_pkts_lenght'",
              5: "(A2.d(2)) Pearson Coefficient 'Avg_pkts_lenght' x 'Avg_payload'",
              6: "(A2.d(3)) Linear Regression model predicting “Avg_pkts_lenght”",
              7: "(A2.e) Number of entries in 'Label' column ",
              8: "(B1) Shuffle the rows of the data framework",
              9: "(B2) Normalize the feature values using min-max",
              10: "(B3,B4) Split data for features and label and create the training sets",
              11: "(B5) K-NN Classification, maximized k and confusion matrix",
              12: "(B6) 5-fold validation",
              13: "(B7) F1 score, precision and results",
              14: "(B8) Comparison",
              15: "Exit"}

# 1. Load the data file (‘dataOS.csv’) into a data frame.
data = pd.read_csv("dataOs.csv")
norm_data = data
global train_df, train_label, test_df, test_label, df_label, red_data
pd.set_option('display.max_columns', None)


def select_Number():
  while True:
    try:
      return int(input("Choose the solution by selecting a number corresponding to Project's PDF:"))
      break
    except:
      print("Please write a valid integer number")

def selection_screen():
  line()
  for num in case_study:
    print(num, ":", case_study[num])
  keystroke = select_Number()
  if keystroke in case_study and keystroke != 15:
    solutions[keystroke]()
    selection_screen()
  elif keystroke == 15:
    quit()
  else:
    print("Please choose a valid number", "\n")
    selection_screen()

def A2a():
  print(data.describe())

def A2b():
  plt.plot(data["Duration_window_flow"], data["Avg_pkts_lenght"], 'o', color="black", markersize=0.5)
  plt.xlabel("Duration_window_flow")
  plt.ylabel("Avg_pkts_lenght")
  plt.show()

def A2c():
  plt.boxplot(data["Avg_pkts_lenght"])
  plt.ylabel("Avg_pkts_lenght")
  plt.show()

def A2d1():
  print(np.corrcoef(data["Duration_window_flow"], data["Avg_pkts_lenght"]))

def A2d2():
  print(np.corrcoef(data["Avg_pkts_lenght"], data["Avg_payload"]))

def A2d3():
  model = LinearRegression()
  x = np.reshape(data["Avg_payload"].to_numpy(), (-1, 1))
  model.fit(x, data["Avg_pkts_lenght"])
  print(model.coef_)
  model.score(x, data["Avg_pkts_lenght"])
  plt.xlabel("Avg_payload")
  plt.ylabel("Avg_pkts_lenght")
  y_predict = model.predict(x)
  plt.plot(data["Avg_payload"], data["Avg_pkts_lenght"], "o", color="blue", markersize=0.1)
  plt.plot(data["Avg_payload"], y_predict, "-", color="red", markersize=0.9)
  plt.show()

def A2e():
  print(data.groupby("Label").count())

def B1():
  global norm_data
  norm_data = norm_data.sample(frac=1).reset_index(drop=True)
  print(norm_data.head(5))

def B2():
  global norm_data, red_data
  scaler = MinMaxScaler()
  X_nr = norm_data[["Avg_syn_flag", "Avg_fin_flag", "Avg_ack_flag", "Avg_psh_flag",
                    "Avg_rst_flag", "Avg_DNS_pkt", "Avg_TCP_pkt", "Avg_UDP_pkt", "Avg_ICMP_pkt",
                    "Duration_window_flow", "Avg_delta_time", "Min_delta_time", "Max_delta_time",
                    "StDev_delta_time", "Avg_pkts_lenght", "Min_pkts_lenght", "Max_pkts_lenght",
                    "StDev_pkts_lenght", "Avg_small_payload_pkt", "Avg_payload", "Min_payload",
                    "Max_payload", "StDev_payload", "Avg_DNS_over_TCP"]]
  scaler.fit(X_nr)
  X_nr = scaler.transform(X_nr)
  norm_data[["Avg_syn_flag", "Avg_fin_flag", "Avg_ack_flag", "Avg_psh_flag",
             "Avg_rst_flag", "Avg_DNS_pkt", "Avg_TCP_pkt", "Avg_UDP_pkt", "Avg_ICMP_pkt",
             "Duration_window_flow", "Avg_delta_time", "Min_delta_time", "Max_delta_time",
             "StDev_delta_time", "Avg_pkts_lenght", "Min_pkts_lenght", "Max_pkts_lenght",
             "StDev_pkts_lenght", "Avg_small_payload_pkt", "Avg_payload", "Min_payload",
             "Max_payload", "StDev_payload", "Avg_DNS_over_TCP"]] = X_nr
  print(norm_data.head(5))
  red_data = norm_data
  red_data.drop(labels=["Avg_syn_flag", "Avg_fin_flag", "Avg_rst_flag", "Avg_DNS_pkt",
                        "Avg_ICMP_pkt", 'Min_payload', "Avg_DNS_over_TCP", "Min_delta_time",
                        "Max_payload", "StDev_delta_time", "Max_pkts_lenght", "Avg_delta_time",
                        "Duration_window_flow", "Max_delta_time", "Min_pkts_lenght", "Avg_psh_flag"], axis=1, inplace=True)

def B3_B4():
  global norm_data, train_df, train_label, test_df, test_label, df_label
  df_label = norm_data["Label"]
  norm_data.drop(labels=["Label"], axis=1, inplace=True)
  training_length = 2000
  train_df = norm_data.iloc[:training_length]
  test_df = norm_data.iloc[training_length:]
  train_label = df_label.iloc[:training_length]
  test_label = df_label.iloc[training_length:]
  print(len(train_df), len(train_label), len(test_df), len(test_label))

def B5():
  # global train_df, train_label, test_df, test_label
  record = {}
  for i in range(1, 637):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(test_df, test_label)
    record[i-1] = clf.score(test_df, test_label)
  print(record)
  clf = KNeighborsClassifier(n_neighbors=1)
  clf.fit(test_df, test_label)
  test_predict = clf.predict(test_df)
  cm = confusion_matrix(test_label, test_predict)
  print(cm)

def B6():
  global norm_data, train_df, train_label, test_df, test_label, df_label
  clf = KNeighborsClassifier(n_neighbors=1)
  kf = KFold(n_splits=5)
  all_scores = []
  for train_index, test_index in kf.split(norm_data):
    train_df = norm_data.iloc[train_index]
    test_df = norm_data.iloc[test_index]
    train_label = df_label.iloc[train_index]
    test_label = df_label.iloc[test_index]
    clf.fit(train_df, train_label)
    score = clf.score(test_df, test_label)
    score = round(score, 4)
    all_scores.append(score)
  print(all_scores)
  quit()

def B7():
  clf = KNeighborsClassifier(n_neighbors=1)
  clf.fit(train_df, train_label)
  test_predict = clf.predict(test_df)
  f1 = f1_score(test_label, test_predict, pos_label="Windows10")
  precision = precision_score(test_label, test_predict, pos_label="Windows10")
  recall = recall_score(test_label, test_predict, pos_label="Windows10")
  print("f1 score is: ", f1)
  print("Precision score is: ", precision)
  print("Recall score is: ", recall)
  quit()

def B8():
  global red_data
  df_label = red_data["Label"]
  red_data.drop(labels=["Label"], axis=1, inplace=True)
  training_length = 2000
  train_df = red_data.iloc[:training_length]
  test_df = red_data.iloc[training_length:]
  train_label = df_label.iloc[:training_length]
  test_label = df_label.iloc[training_length:]
  print(len(train_df), len(train_label), len(test_df), len(test_label))
  clf = KNeighborsClassifier(n_neighbors=1)
  kf = KFold(n_splits=5)
  all_scores = []
  for train_index, test_index in kf.split(red_data):
    train_df = red_data.iloc[train_index]
    test_df = red_data.iloc[test_index]
    train_label = df_label.iloc[train_index]
    test_label = df_label.iloc[test_index]
    clf.fit(train_df, train_label)
    score = clf.score(test_df, test_label)
    score = round(score, 4)
    all_scores.append(score)
  print(all_scores)
  quit()

def main():
  selection_screen()

def line():
  print("- - - - - - - - - - - - - - - - - - - - - - - - "
        "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")


solutions = {1: A2a, 2: A2b, 3: A2c, 4: A2d1,
             5: A2d2, 6: A2d3, 7: A2e, 8: B1, 9: B2,
             10: B3_B4, 11: B5, 12: B6, 13: B7, 14: B8}

if __name__ == "__main__":
  main()