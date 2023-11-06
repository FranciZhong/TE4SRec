import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def log_eval_top_k(top_k, eval_top_k, mrr, path):
    eval_columns = []
    column_headers = []
    for i, k in enumerate(top_k):
        ndcg, hit_rate = eval_top_k[i]
        hit_rate = round(hit_rate * 100, 4)
        ndcg = round(ndcg * 100, 4)
        if k == 1:
            column_headers.append('HR@1')
            eval_columns.append([hit_rate])
        else:
            column_headers.append(f'HR@{k}')
            eval_columns.append([hit_rate])
            column_headers.append(f'NDCG@{k}')
            eval_columns.append([ndcg])

    column_headers.append('MRR')
    eval_columns.append([round(mrr * 100, 4)])

    df = pd.DataFrame(zip(*eval_columns), columns=column_headers)
    df.to_csv(path, index=False)


def log_eval_history(top_k, eval_valid_results, mrr_results, final_epoch, output_path):
    eval_columns = []
    column_headers = []
    for i, k in enumerate(top_k):
        hit_rate_list = [round(hit_rate * 100, 4) for ndcg, hit_rate in eval_valid_results[i]]
        # let it start with 0
        hit_rate_list.insert(0, 0.0)
        ndcg_list = [round(ndcg * 100, 4) for ndcg, hit_rate in eval_valid_results[i]]
        ndcg_list.insert(0, 0.0)
        if k == 1:
            column_headers.append('HR@1')
            eval_columns.append(hit_rate_list)
        else:
            column_headers.append(f'HR@{k}')
            eval_columns.append(hit_rate_list)
            column_headers.append(f'NDCG@{k}')
            eval_columns.append(ndcg_list)

    mrr_results.insert(0, 0.0)
    column_headers.append('MRR')
    eval_columns.append([round(mrr * 100, 4) for mrr in mrr_results])

    df = pd.DataFrame(zip(*eval_columns), columns=column_headers)[:final_epoch + 1]
    df.to_csv(output_path + '/top_k_tables.csv', index=False)

    sns.set_theme(style='darkgrid')

    sns.lineplot(data=df)

    plt.savefig(output_path + '/eval_valid_line.png')
    plt.clf()


def print_signals(signal_arr, output_path):
    for signal in signal_arr:
        sns.lineplot(signal)

    plt.xlabel('Temporal points')
    plt.ylabel('Signals')

    file = 'temporal_encoding.png'
    file = output_path + file if output_path.endswith('/') else output_path + '/' + file
    plt.savefig(file)
    plt.clf()
