import glob
import pandas as pd
import matplotlib.pyplot as plt

def load_loss(path): 
    files = glob.glob(path)
    dfs_split = []
    for file in files: 
        df = pd.read_csv(file, index_col=[0,1], header=[0,1])
        dfs_split.append(df)
    
    return dfs_split

if __name__ == '__main__':
    path = '../result/gefcom2014-solar/trial21/dfs_loss_valid_model/*.csv'
    dfs_loss_split_valid = load_loss(path)
    loss_mean_valid = [df.mean().mean() for df in dfs_loss_split_valid]

    # Load competition scores
    df_scores = pd.read_excel('../data/gefcom2014/gefcom2014-scores.xlsx', index_col=0, header=0, sheet_name='Solar')
    df_scores.loc['Solar trial',:] = loss_mean_valid
    df_scores.loc[:,'Overall'] = df_scores.mean(axis=1)

    # Plot scores for teams
    ax = df_scores['Overall'].sort_values().plot.bar(figsize=(15,5))
    ax.set_ylabel('pinball loss', fontsize=14)
    plt.savefig('./gefcom2014-solar-teams.png')

    # Plot scores for tasks
    fig, ax = plt.subplots(figsize=(15,5))
    df_scores.loc['Gang-gang',:][:-1].plot(ax=ax)
    df_scores.loc['dmlab',:][:-1].plot(ax=ax)
    df_scores.loc['C3 Green Team',:][:-1].plot(ax=ax)
    df_scores.loc['Solar trial',:][:-1].plot(ax=ax)
    ax.set_xlabel('task', fontsize=14)
    ax.set_ylabel('pinball loss', fontsize=14)
    ax.legend()
    plt.savefig('./gefcom2014-solar-tasks.png')
