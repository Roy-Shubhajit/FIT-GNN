import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import latexipy as lp

# lp.latexify()

df = pd.read_csv("results.csv") 
for dataset in df["dataset"].unique():
    with lp.figure(filename=f"{dataset}_sum_squares_subgraph_nodes", directory="plots", size = (6, 4), exts=["pdf"]):
        plt.plot(df[df["dataset"] == dataset]["coarsening_ratio"], np.log10(df[df["dataset"] == dataset]["sum_squares_subgraph_nodes"]), label = 'Ours', marker = "o", color = 'b')
        plt.plot(df[df["dataset"] == dataset]["coarsening_ratio"], np.log10(df[df["dataset"] == dataset]["tot_num_orig_nodes_squared"]), label = 'Original', marker = "^", color = 'g' )
        plt.title(f"{dataset}\nSum of squares of subgraph nodes vs Coarsening Ratio" )
        plt.xlabel("Coarsening Ratio")
        plt.ylabel("Sum of squares of subgraph nodes (log scale)")
        plt.grid(alpha = 0.5)
        plt.legend()
        lp.save_figure(filename=f"{dataset}_sum_squares_subgraph_nodes", directory="plots", exts = ["pdf"])

for dataset in df["dataset"].unique():
    with lp.figure(filename=f"{dataset}_total_space", directory="plots", size = (6, 4), exts=["pdf"]):
        plt.plot(df[df["dataset"] == dataset]["coarsening_ratio"], np.log10(df[df["dataset"] == dataset]['x'] + df[df["dataset"] == dataset]['y'] + df[df["dataset"] == dataset]['edge_index']), marker = "o", color = 'b')
        # plt.plot(df[df["dataset"] == dataset]["coarsening_ratio"], np.log10((df[(df["dataset"] == dataset) & (df["baseline"]==True)]['x'] + df[(df["dataset"] == dataset) & (df["baseline"]==True)]['y'] + df[(df["dataset"] == dataset) & (df["baseline"]==True)]['edge_index']).iloc[0] * np.ones(7)), marker = "^", color = 'g')
        # plt.legend(["Ours", "Original"])
        plt.title(f"{dataset} - Total space occupied* vs Coarsening Ratio\n*Not including sizes of train, val and test masks")
        plt.xlabel("Coarsening Ratio")
        plt.ylabel("Total space occupied (in log10(byes))")
        plt.grid(alpha = 0.5)
        lp.save_figure(filename=f"{dataset}_total_space", directory="plots", exts = ["pdf"])
