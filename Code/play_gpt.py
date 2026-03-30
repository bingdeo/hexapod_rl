import logging 
import os
import subprocess
import argparse

EUREKA_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../../IsaacLab"
LOG_LOOT_DIR = 'D:/Eureka logs'

parser = argparse.ArgumentParser(description="Play a log")
parser.add_argument("--task",type=str,default='060401',help='The folder number of task log')
parser.add_argument("--iter",type=int,default=0,help='The iteration number of task file')
parser.add_argument("--sample",type=int,default=0,help='The sample number of task file')
parser.add_argument("--model",type=int,default=0,help='The model number to play')
args = parser.parse_args()


def main():

    task_id = 'Hexapod-Play-v0'
    num_envs = 300
    suffix = '_gpt'

    task_num = args.task
    iteration_num = args.iter
    sample_num = args.sample
    model_num = args.model

    task_file = f'{EUREKA_ROOT_DIR}/../../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/hexapod_play/hexapod_env.py'
    task_code_string  = open(task_file, encoding='utf-8').read()
    gpt_file = f'D:/Eureka logs/{task_num}/env_iter{iteration_num}_response{sample_num}_rewardonly.py'
    gpt_code_string = open(gpt_file, encoding='utf-8').read()
    output_file = f"{ISAAC_ROOT_DIR}/source/isaaclab_tasks/isaaclab_tasks/direct/hexapod_play/hexapod_env{suffix.lower()}.py"
    
    log_file = f'D:/Eureka logs/{task_num}/env_iter{iteration_num}_response{sample_num}.txt'
    log_string = open(log_file, encoding='utf-8').read()

    lines = log_string.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('Exact experiment'):
            break
    task_logdir = line.split(':')[-1].strip()

    model = f'D:/Eureka logs/{task_num}/logs/{task_logdir}/model_{model_num}.pt'


    lines = gpt_code_string.split("\n")
    try:
        keys_index = lines.index("            #keys")
    except:
        logging.info("#keys error")

    reward_signature = lines[0:keys_index]
    keys = lines[keys_index:]


    reward_signature = "\n".join([line for line in reward_signature])
    if "#rewards_part" in task_code_string:
        task_code_string_iter = task_code_string.replace("#rewards_part", "#rewards_part\n" + reward_signature)
    else:
        logging.info("No #rewards_part")
        raise NotImplementedError

    keys = "\n".join([line for line in keys])
    if "#keys_part" in task_code_string:
        task_code_string_iter = task_code_string_iter.replace("#keys_part", "#keys_part\n" + keys)
    else:
        raise NotImplementedError

    with open(output_file, 'w') as file:
        file.writelines(task_code_string_iter + '\n')

    
        
    process = subprocess.Popen(['C:/Users/Administrator/IsaacLab/isaaclab.bat', '-p',
                                'C:/Users/Administrator/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py',
                                f'--task={task_id}',
                                f'--num_envs={num_envs}',
                                f'--checkpoint={model}',
                                ],
                                stdout=None, stderr=None)

    process.communicate()


if __name__ == "__main__":
    main()