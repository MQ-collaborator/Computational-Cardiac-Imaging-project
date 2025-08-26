import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from conventional_utils import home_directory

sorted_coefficients_path = home_directory / "data" / "regression_results" / "sorted_coefficients_f.csv"

def main(font = 16):
    df = pd.read_csv(sorted_coefficients_path)
    
    for _, row in df.iterrows():
        plt.errorbar(x=row['Coefficient'], y=row['Column'],
                                         #xerr=[[row['metabolite beta'] - row['CI 0.025']], [row['CI 0.975'] - row['metabolite beta']]],
                                         fmt='o', markersize=7, label=row['Coefficient'], color='tab:blue', elinewidth=3)
        #errors = f'{np.round(row["metabolite beta"], 2)} ({np.round(row["CI 0.025"], 3)}, {np.round(row["CI 0.975"], 3)})'
                # plt.text(row['metabolite beta'] * 0 + 0.81, row['metabolite'],
                #                  errors,
                #                  ha='left', va='center', size=14, style='normal', color='black')
        plt.axvline(x=0, color='black', linestyle='dashed', alpha=0.6)
        plt.xlabel('Beta Coefficient', fontsize=font)
        # plt.grid(which='major', color='#EBEBEB', linewidth=0.8)
        plt.xlim(-0.5, 0.5)
        plt.xticks([-1, 0, 1], fontsize=font)
        plt.yticks(fontsize=font)
        plt.gca().set_axisbelow(True)
        plt.gca().margins(y=0.03)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        ax = plt.gca()
        ax.tick_params(length = 0)
        plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()