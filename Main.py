import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

case_study = {1: "Descriptive Statistics(A2(a))", 2: "Empty", 3: "Empty", 5: "Exit"}

# 1. Load the data file (‘dataOS.csv’) into a data frame.
data = pd.read_csv("dataOs.csv")


def select_Number():
  while True:
    try:
      return int(input("Choose the solution by selecting a number corresponding to Project's PDF:"))
      break
    except:
      print("Please write a valid integer number")

def selection_screen():
  for num in case_study:
    print(num, ":", case_study[num])
  keystroke = select_Number()
  if keystroke in case_study and keystroke != 5:
    solutions[keystroke]()
    selection_screen()
  elif keystroke == 5:
    quit()
  else:
    print("Please choose a valid number", "\n")
    selection_screen()

def A2a():
  columns = data.columns.values
  print("Dimensionality:", data.shape, "First 10 rows:", data.head(10), "\n", "Columns descriptive statistic")
  for i in range(1, columns.size - 1):
    temp = data[columns[i]]
    print(columns[i], ": mean =", temp.mean(), ", max =", temp.max(), ", min =", temp.min(), ", std =", temp.std())

def main():
  selection_screen()


solutions = {1: A2a, 2: "Nonenytime"}

if __name__ == "__main__":
  main()
