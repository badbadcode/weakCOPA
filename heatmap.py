import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path = "casestudy/word_importance.csv"
df = pd.read_csv(path, index_col=2, header=0)
correct_df = df[df["names"] == "correct"]
plot_df = correct_df.loc[:,["db-l", "db-l-reg"]]
grid_kws = {"height_ratios": (.2, .01), "hspace": 0}
f, (ax, cbar_ax) = plt.subplots(2, figsize=(4.7,3.5),gridspec_kw=grid_kws,)
df_t = pd.DataFrame(plot_df.values.T, index=plot_df.columns, columns=plot_df.index)  # transpose

ax = sns.heatmap(df_t, ax=ax,
                 cbar_ax=cbar_ax,
                 cmap='Greys',
                 center=0,linewidths=0.3,square=True,
                 cbar_kws={"orientation": "horizontal"})
plt.show()
plt.savefig("casestudy/word_importance.png",dpi=720,bbox_inches = 'tight') # for the clear PNG