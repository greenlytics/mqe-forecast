import glob
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt

def load_loss(path): 
    files = glob.glob(path)
    files = natsorted(files)
    dfs_split = []
    for file in files: 
        df = pd.read_csv(file, index_col=[0,1], header=[0,1])
        dfs_split.append(df)
    
    return dfs_split

if __name__ == '__main__':
    path = './results/gefcom2014-solar/trial45/dfs_loss_valid/*.csv'
    dfs_loss_split_valid = load_loss(path)
    loss_mean_valid = [df.mean().mean() for df in dfs_loss_split_valid]

    # Load competition scores
    df_scores = pd.read_excel('./data/gefcom2014/gefcom2014-scores.xlsx', index_col=0, header=0, sheet_name='Solar')
    df_scores.index = ['Participant {0}'.format(i+1) for i in range(len(df_scores.index))]
    df_scores.loc['Solar trial',:] = loss_mean_valid
    df_scores.loc[:,'Overall'] = df_scores.iloc[:, 4:].mean(axis=1)
    df_scores = df_scores.rename(index={'Participant 3': 'Benchmark'})

    # Plot scores for teams
    ax = df_scores['Overall'].sort_values().plot.bar(figsize=(15,5))
    ax.set_ylabel('pinball loss', fontsize=14)
    plt.tight_layout()
    plt.savefig('./plots/gefcom2014-solar-teams.png')

    # Plot scores for tasks
    fig, ax = plt.subplots(figsize=(15,5))
    fig, ax = plt.subplots(figsize=(15,5))
    df_scores.loc['Participant 6',:][:-1].plot(ax=ax)
    df_scores.loc['Participant 20',:][:-1].plot(ax=ax)
    df_scores.loc['Participant 4',:][:-1].plot(ax=ax)
    ylim = ax.get_ylim()
    df_area = pd.DataFrame(data=np.repeat(ylim[1], 4), index=df_scores.columns[:4], columns=['Testing period'])
    df_area.plot.area(ax=ax, color='grey', alpha=0.3)
    df_scores.loc['Solar trial',:][:-1].plot(ax=ax)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel('task', fontsize=14)
    ax.set_ylabel('pinball loss', fontsize=14)
    ax.grid()
    ax.legend()
    plt.savefig('./plots/gefcom2014-solar-tasks.png')
