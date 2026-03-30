import numpy as np 
import json
import logging 
import os
import openai
import re
import subprocess
from pathlib import Path
import shutil
import time 

from utils.misc import * 
from utils.file_utils import load_tensorboard_logs
from utils.extract_task_code import *

EUREKA_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../../IsaacLab"

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True
)

def main():

    iteration = 6
    temperature = 0.8
    max_iterations = 1000
    sample = 4
    task_id = 'Hexapod-Eureka-v0'
    num_envs = 300

    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    task = "hexapod"
    task_description = "Make the hexapod run forward stably"
    suffix = "_gpt"
    model = "gpt-4o"
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    task_file = f'{EUREKA_ROOT_DIR}/../../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/hexapod_gpt/hexapod_env.py'
    task_code_string  = open(task_file, encoding='utf-8').read()
    task_obs_file = f'{EUREKA_ROOT_DIR}/../../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/hexapod_gpt/hexapod_obs.py'
    task_obs_code_string = open(task_obs_file, encoding='utf-8').read()
    output_file = f"{ISAAC_ROOT_DIR}/source/isaaclab_tasks/isaaclab_tasks/direct/hexapod_gpt/hexapod_env{suffix.lower()}.py"

    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'
    initial_system = open(f'{prompt_dir}/initial_system.txt', encoding='utf-8').read()
    code_output_tip = open(f'{prompt_dir}/code_output_tip.txt', encoding='utf-8').read()
    code_feedback = open(f'{prompt_dir}/code_feedback.txt', encoding='utf-8').read()
    initial_user = open(f'{prompt_dir}/initial_user.txt', encoding='utf-8').read()
    reward_signature = open(f'{prompt_dir}/reward_signature.txt', encoding='utf-8').read()
    policy_feedback = open(f'{prompt_dir}/policy_feedback.txt', encoding='utf-8').read()
    execution_error_feedback = open(f'{prompt_dir}/execution_error_feedback.txt', encoding='utf-8').read()

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]
    
    """
    logging.info(f"System:{initial_system}")
    logging.info(f"User:{initial_user}")
    """

    DUMMY_FAILURE = -10000.
    max_successes = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    
    # Eureka generation loop
    for iter in range(iteration):
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = sample if "gpt-4o" in model else 4

        logging.info(f"Iteration {iter}: Generating {sample} samples with {model}")

        while True:
            if total_samples >= sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = openai.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        n=chunk_size
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    logging.error(f"[OpenAI]  {type(e).__name__}: {e}")
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur.choices)
            prompt_tokens = response_cur.usage.prompt_tokens
            total_completion_token += response_cur.usage.completion_tokens
            total_token += response_cur.usage.total_tokens

        if sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0].message.content + "\n")

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        
        code_runs = [] 
        rl_runs = []
        for response_id in range(sample):
            response_cur = responses[response_id].message.content
            logging.info(f"[RAW GPT] iter {iter} sample {response_id} ===\n{response_cur}\n")
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            try:
                keys_index = lines.index("            #keys")
            except:
                logging.info("#keys error")
                continue

            reward_signature = lines[0:keys_index]
            keys = lines[keys_index:]

            code_runs.append(code_string)

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

            # Save the new environment code when the output contains valid code string!
            with open(output_file, 'w') as file:
                file.writelines(task_code_string_iter + '\n')

            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            # Find the freest GPU to run GPU-accelerated RL
            set_freest_gpu()
            
            # Execute the python file with flags
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w', encoding="utf-8") as f:
                process = subprocess.Popen(['C:/Users/Administrator/IsaacLab/isaaclab.bat', '-p',
                                            'C:/Users/Administrator/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py',
                                            f'--task={task_id}',
                                            f'--num_envs={num_envs}',
                                            f'--max_iterations={max_iterations}',
                                            f'--headless'],
                                            stdout=f, stderr=f)
            rl_runs.append(process)
            time.sleep(2)


        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        successes = []
        code_paths = []
        
        exec_success = False 
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_run.communicate()
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            try:
                with open(rl_filepath,  encoding="utf-8") as f:
                    stdout_str = f.read() 
            except: 
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content) 
                successes.append(DUMMY_FAILURE)
                continue

            content = ''

            
            traceback_msg = filter_traceback(stdout_str)
            logging.info(f'Traceback Message:\n {traceback_msg}')

            # If there is no error
            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Tensorboard Directory:'):
                        logging.info(line)
                        break
                tensorboard_logdir = line.split(':')[-1].strip()
                logging.info(f'tensorboard_logdir:{tensorboard_logdir}')
                tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                logging.info(f'tensorboard_logs:{tensorboard_logs}')
                max_iterations = np.array(tensorboard_logs['Episode_Termination/time_out']).shape[0]
                logging.info(f"max_iterations:{max_iterations}")
                epoch_freq = max(int(max_iterations // 10), 1)
                
                content += policy_feedback.format(epoch_freq=epoch_freq)
                
                if "Episode_Reward/total" in tensorboard_logs:
                    total_reward = np.array(tensorboard_logs["Episode_Reward/total"])

                # Add reward components log to the feedback
                for metric in tensorboard_logs:
                    if "Episode_Reward/" in metric:
                        logging.info(f"Metric:{metric}")
                        metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                        metric_cur_max = max(tensorboard_logs[metric])
                        logging.info(f"Metric_cur_max:{metric_cur_max}")
                        metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                        logging.info(f"Metric_cur_mean:{metric_cur_mean}")
                        if "Episode_Reward/total" == metric:
                            successes.append(metric_cur_max)
                        metric_cur_min = min(tensorboard_logs[metric])
                        if metric != "total_reward":
                            if metric != "Episode_Reward/total":
                                metric_name = metric 
                            else:
                                metric_name = "task_score"
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                        else:
                            # Provide ground-truth score when success rate not applicable
                            if "Episode_Reward/total" not in tensorboard_logs:
                                content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                code_feedbacks.append(code_feedback)
                content += code_feedback  
            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content) 
        
        # Repeat the iteration if all code generation failed
        if not exec_success and sample != 1:
            execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue

        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]

        max_success = successes[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.) / sample

        # Update the best Eureka Output
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx].message.content + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        # Save dictionary as JSON file
        with open('messages.json', 'w') as file:
            json.dump(messages, file, indent=4)

if __name__ == "__main__":
    main()