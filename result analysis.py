import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 16})

# file1 = 'results_default_baseline.txt'
# file2 = 'results_close_level.txt'
# file3 = 'results_default_selfadjust_delete58.txt'
folder = 'human_3,4,5/'
file = folder + 'results_adjust.txt'
# file = folder + 'results_fix_4.txt'
# file2 = 'results_eps0.3_mix.txt'
# files = []
# for r, d, f in os.walk('test_the_level/'):
#     for file in f:
#         if '.txt' in file:
#             files.append(os.path.join(r, file))

output_grade = folder + 'delta_grade.png'
output_dur = folder + 'duration.png'
output_avg_level = folder + 'avg_agent_level.png'

output_grade_box = folder + 'delta_grade_box.png'
output_dur_box = folder + 'duration_box.png'

# titlename = 'Comparison of baseline and self-adjustment (default environment)'
# titlename = 'new_test'

def get_number(filename):
    num = filename[-5]
    return np.int(num)

def update_round_info(round_info, iter, max_win, win_or_lose, final_win, duration, agent_levels, avg_agent_level, human_level):
    round_info['iter'].append(iter)
    round_info['max_win'].append(max_win)
    round_info['win_or_lose'].append(win_or_lose)
    round_info['final_win'].append(final_win)
    round_info['duration'].append(duration)
    round_info['agent_levels'].append(agent_levels)
    round_info['avg_agent_level'].append(avg_agent_level)
    round_info['human_level'].append(human_level)
    return round_info

round_infos = []
test_names = []
# files = np.sort(files)[::-1]

# for file in files:
#     num = get_number(file)
#     test_names.append('Level ' + np.str(num))
test_names.append('fix')

files = [file]
# files = [file2]
# files = np.append(files, [file_adjust])

for j in range(len(files)):
    round_info = {"test": test_names[j],
                  "iter": [],
                  "max_win": [],
                  "win_or_lose": [],
                  "final_win": [],
                  "duration": [],
                  "agent_levels": [],
                  "avg_agent_level": [],
                  "human_level": []}
    file = files[j]
    i = 0
    round_cnt = 0
    with open(file, 'r') as f:
        for line in f:
            if i % 3 == 0:
                agent_levels = line.split(',')
                agent_levels = [int(i) for i in agent_levels]
                avg_agent_level = np.average(agent_levels)

            elif i % 3 == 1:
                win_or_lose = line.split(',')        # +1 agent win, -1 human win
                win_or_lose = [int(float(i)) for i in win_or_lose]

                # max absolute difference of grades within one round
                max_win = max(np.cumsum(win_or_lose), key=abs)

            elif i % 3 == 2:
                lose, win, duration, human_level = line.split(',')
                lose, win, duration, human_level = int(lose), int(win), int(duration), float(human_level)

                # difference of grades at the end of one round
                final_win = win - lose

                # update round_info
                round_info = update_round_info(round_info, np.int((i + 1)/3) ,max_win, win_or_lose, final_win, duration, agent_levels, avg_agent_level, human_level)

            i += 1
            # if np.int(i / 3) >= 50:
            #     break
    round_infos.append(round_info)
    print(round_info)

# plot grade difference
# plt.title(titlename)
plt.figure(figsize=(7, 5))
# plt.title('Score Gap After One Round')
plt.plot(round_info['iter'], np.zeros_like(round_info['iter']), 'gray',linestyle="--", linewidth=2)
# plt.plot(round_infos[0]['iter'], round_infos[0]['final_win'], color='darkred', linewidth=2, label=round_infos[0]['test'][0])
# plt.plot(round_infos[1]['iter'], round_infos[1]['final_win'], color='purple', linewidth=2, label=round_infos[1]['test'][0])
# plt.plot(round_infos[2]['iter'], round_infos[2]['final_win'], color='blue', linewidth=2, label=round_infos[2]['test'][0])
plt.plot(round_info['iter'], round_info['final_win'], color='purple', linewidth=2, label='score gap after one round')
plt.plot(round_info['iter'], round_info['max_win'], color='green', linestyle=":", linewidth=2, label='max score gap within one round')
plt.xlabel('Round')
plt.ylabel('Score Gap')
plt.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig(output_grade)
plt.close()


# plot duration
plt.figure(figsize=(7, 5))
# plt.title(titlename)
# plt.title('Duration of One Round')
# plt.plot(round_infos[0]['iter'], round_infos[0]['duration'], color='darkred', linewidth=2, label=round_infos[0]['test'][0])
# plt.plot(round_infos[1]['iter'], round_infos[1]['duration'], color='purple', linewidth=2, label=round_infos[1]['test'][0])
# plt.plot(round_infos[2]['iter'], round_infos[2]['duration'], color='blue', linewidth=2, label=round_infos[2]['test'][0])
plt.plot(round_info['iter'], round_info['duration'], color='darkred', linewidth=2, label='Duration of one round')
plt.xlabel('Round')
plt.ylabel('Duration')
# plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(output_dur)
plt.close()


# # plot average level in one round
# plt.title(titlename)
# plt.plot(round_info['iter'], round_info['avg_human_level'], color='blue', linewidth=2, label='Average level of RL agent in one round')
# plt.xlabel('Round')
# plt.ylabel('Average level of RL agent')
# plt.legend()
# plt.savefig(output_avg_level)
# plt.close()


# # plot agent level and score gap in a certain round
# # random_round_idx = np.random.randint(0, len(round_info['iter']), size=5)
# random_round_idx = range(len(round_info['agent_levels']))
# for idx in random_round_idx:
#     agent_levels = round_info['agent_levels'][idx]
#     score_gap = np.cumsum(round_info['win_or_lose'][idx])
#
#     fig, ax1 = plt.subplots()
#
#     color = 'tab:red'
#     ax1.set_xlabel('Step')
#     ax1.set_ylabel('Score Gap', color=color)
#     ax1.plot(range(1, len(score_gap)+1), score_gap, color=color)
#     ax1.tick_params(axis='y', labelcolor=color)
#     ax1.plot(range(1, len(score_gap)+1), np.zeros_like(score_gap), 'mistyrose', linestyle="--", linewidth=2)
#     ax1.set_ylim([-np.max(np.abs(score_gap))-1, np.max(np.abs(score_gap))+1])
#
#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
#     color = 'tab:blue'
#     ax2.set_ylabel('Agent Level', color=color)  # we already handled the x-label with ax1
#     ax2.plot(range(1, len(agent_levels)+1), agent_levels, color=color)
#     ax2.tick_params(axis='y', labelcolor=color)
#     ax2.set_ylim([0, 9])
#
#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     plt.show()



# plot boxplot grade difference
# plt.title('Comparison of the score gap among different settings')

# plt.title('two agents eps = 0.3')
# grad_data = [info['final_win'] for info in round_infos]
# # grad_data = [round_infos[0]['final_win'], round_infos[1]['final_win'], round_infos[2]['final_win']]
# plt.boxplot(grad_data, showfliers=True)
# plt.plot(range(12), np.zeros(12), 'gray',linestyle="--", linewidth=1.5)
#
# plt.xticks(range(1,len(files)+1), test_names)
# # plt.xticks(rotation=-45)
# plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))
# plt.ylabel('Score Gap')
#
# plt.tight_layout()
# plt.savefig(output_grade_box)
# plt.close()


# print(np.average(round_infos[0]['final_win']))
# print(np.average(round_infos[1]['final_win']))



#
#
# # plot boxplot duration
# # plt.title('Comparison of longing duration among different settings')
# grad_data = [info['duration'] for info in round_infos]
# # grad_data = [round_infos[0]['duration'], round_infos[1]['duration'], round_infos[2]['duration']]
# plt.boxplot(grad_data, showfliers=True)
#
# plt.xticks(range(1,len(files)+1), test_names)
# plt.xticks(rotation=-45)
# plt.gca().yaxis.set_major_locator(plt.MultipleLocator(250))
# plt.ylabel('Longing Duration')
#
# plt.tight_layout()
# plt.savefig(output_dur_box)
# plt.close()


# statistics of the results
print('=====score gap=====')
print('mean: ', np.mean(round_info['final_win']))
print('median: ', np.median(round_info['final_win']))
print('std: ', np.std(round_info['final_win']))


print('=====duration=====')
print('mean: ', np.mean(round_info['duration']))
print('median: ', np.median(round_info['duration']))
print('std: ', np.std(round_info['duration']))