import time
import gym
import os
import cv2
import numpy as np
from collections import deque
import torch
import argparse
import sys
sys.path.append('/home/wuk/Instruction_Aware_Tracking/')

from agent_CNN_LSTM_GoalCondition_bbox import CQLSAC_CNN_LSTM_GoalCondition_bbox
from buffer_GoalCondition import ReplayBuffer


from utils import save, collect_random, evaluate, de_normalize, rgb_to_binary_mask, get_bounding_box, \
    generate_new_bbox_image, normalize_bbox
import random
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE
import re
# from deva import DEVAInferenceCore
# from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
# from deva.ext.grounding_dino import get_grounding_dino_model
# from deva.inference.eval_args import add_common_eval_args, get_model_and_config
# from deva.inference.result_utils import ResultSaver
# from deva.ext.with_text_processor import process_frame_with_text as process_frame

from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
import multiprocessing as mp

from system_prompt import system_prompt_recovery_v1, system_prompt_recovery_v2,system_prompt_recovery_v3,system_prompt_context_v1,encode_image_array,system_prompt_perception
from improving import ExperienceManager
# MiniCPM_model = AutoModel.from_pretrained(
#     'openbmb/MiniCPM-o-2_6',
#     trust_remote_code=True,
#     attn_implementation='sdpa', # sdpa or flash_attention_2
#     torch_dtype=torch.bfloat16,
#     init_vision=True,
#     init_audio=False,
#     init_tts=False
# )
# model = MiniCPM_model.eval().cuda()
# tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

os.environ['UnrealEnv']='/home/wuk/UnrealEnv'

bbox_goal_list = [f for f in os.listdir('../bbox_goal') if f.startswith('robotview_3')]

bbox_goal = [cv2.imread(os.path.join('../bbox_goal', f)) for f in bbox_goal_list]
from PIL import Image
class RecoveryState:
    def __init__(self):
        self.is_recovering = False
        self.failure_imgs = deque(maxlen=3) # observation sampled before failure
        self.recovery_images=[] #observation corresponding to actual executed recovery actions
        self.recovery_actions = [] #vlm generated actions
        self.history_actions = []   #Actual executed actions for recovery (agent may recovery before execute all actions)
        self.trajectory_actions = deque(maxlen=20) #agent actions before failure (used for memory regression)

    def start_recovery(self):
        self.is_recovering = True
        self.recovery_actions = []
        self.history_actions=[]
        self.recovery_images=[]
        
    def set_actions(self, actions):
        self.recovery_actions = actions

    def has_pending_actions(self):
        return len(self.recovery_actions) > 0

    def get_next_action(self,obs):
        action=self.recovery_actions.pop(0)

        if action==[0,200]:
            self.history_actions.append('move forward')
        elif action ==[-40,0]:
            self.history_actions.append('turn left')
        elif action ==[40,0]:
            self.history_actions.append('turn right')
        elif action ==[0,-200]:
            self.history_actions.append('move backward')
        elif 'jump' in action:
            self.history_actions.append('jump over')

        self.recovery_images.append(obs)
        return action
    def complete_recovery(self, success):
        self.is_recovering = False
        self.recovery_actions.clear()



def parse_recovery_analysis_actions(response):
    """Convert VLM response to actual actions"""
    action_map = {
        'forward': ([0, 200], 'move forward'),
        'left': ([-40, 0], 'turn left'),
        'right': ([40, 0], 'turn right'),
        'backward': ([0, -200], 'move backward'),
        'jump': ('jump', 'jump over')
    }
    
    actions = []
    try:
        recovery_analysis=response.lower().split('recovery action')[0]
        if '[context analysis]:' in recovery_analysis:
            recovery_analysis=recovery_analysis.split('[context analysis]:')[1]
        else:
            recovery_analysis = recovery_analysis.split('context analysis')[1]
        action_response = response.lower().split('recovery action')[1].split(':')[1]
        print('analysis:',recovery_analysis)
        recovery_actions_list = action_response.strip('[]').split(',')
        print('action list:',recovery_actions_list)
    except:
        print('Parse error:',response)
        recovery_analysis=None
        recovery_actions_list=[]
    for r_action in recovery_actions_list:
        for key, (action, desc) in action_map.items():
            if key in r_action.lower():
                actions.append(action)
                break
    
    return recovery_analysis,actions

def parse_recovery_action_v2(action_response):
    action_map = {
        'forward': ([0, 200], 'move forward'),
        'left': ([-40, 0], 'turn left'),
        'right': ([40, 0], 'turn right'),
        'backward': ([0, -200], 'move backward'),
        'jump': ('jump', 'jump over')
    }
    actions=[]
    action_response=action_response.lower()
    recovery_actions_list = action_response.strip('[]').split(',')
    for r_action in recovery_actions_list:
        for key, (action, desc) in action_map.items():
            if key in r_action.lower():
                actions.append(action)
                break
    return actions

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-SAC-FlexibleRoom_cnn_lstm_goal_conditioned_Mask_bbox", help="Run name, default: CQL-SAC")
    parser.add_argument("--env", type=str, default="UnrealTrack-Old_Factory_01-ContinuousColorMask-v0", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--buffer_path", type=str,default='E:\FlexibleRoom_Continuous_dataset\multi_discrete_goal_condition_tracktrain_Mask_v0')
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=50, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="")
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="")
    parser.add_argument("--target_action_gap", type=float, default=10, help="")
    parser.add_argument("--with_lagrange", type=int, default=0, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eval_every", type=int, default=100, help="")
    parser.add_argument("--max_distractor", type=int, default=0, help="")
    parser.add_argument("--lstm_seq_len", type=int, default=20, help="")
    parser.add_argument("--lstm_out", type=int, default=64, help="")
    parser.add_argument("--lstm_layer", type=int, default=1, help="")
    parser.add_argument("--input_type", type=str, default='fusion_cnn_lstm', help="")
    parser.add_argument("--load_agent_model", type=str, default='/home/wuk/Instruction_Aware_Tracking/trained_models/server-H20/CQL-SAC-goal_conditioned_DEVA_robot_view_IOUEdgedDisCQL-SAC5000.pth', help="")

    parser.add_argument("--mode", type=str, default='eval', help="")

    # add_common_eval_args(parser)
    # add_ext_eval_args(parser)
    # add_text_default_args(parser)
    # ##init deva
    # deva_model, deva_cfg, args = get_model_and_config(parser)
    # gd_model, sam_model = get_grounding_dino_model(deva_cfg, 'cuda')

    args = parser.parse_args()
    deva_model=None
    deva_cfg=None
    gd_model=None
    sam_model=None
    return args, deva_model, deva_cfg, gd_model, sam_model

def DEVATracker(deva_model, deva_cfg):
    torch.autograd.set_grad_enabled(False)
    deva_cfg['temporal_setting'] = 'online'
    assert deva_cfg['temporal_setting'] in ['semionline', 'online', 'window']
    deva_cfg['enable_long_term_count_usage'] = True
    deva = DEVAInferenceCore(deva_model, config=deva_cfg)
    deva.next_voting_frame = deva_cfg['num_voting_frames'] - 1
    deva.enabled_long_id()
    result_saver = ResultSaver('./deva_out', None, dataset='demo', object_manager=deva.object_manager)
    return result_saver, deva
def DEVAProcess(next_state,deva,sam_model,gd_model,result_saver,deva_step):
    
    next_state_rgb = next_state[0][:, :, 0:3]
    next_state_deva = process_frame(deva, gd_model, sam_model, str(deva_step) + '.jpg', result_saver,
                                            deva_step, image_np=next_state_rgb.astype(np.uint8))
    return next_state_deva

def AgentPolicy(agent,next_state_deva,bbox_goal,ht,ct):
    goal_bbox = bbox_goal[0]
    goal_bbox = cv2.resize(goal_bbox,(160,160))
    next_state_deva=cv2.resize(next_state_deva,(160,160))
    next_state = np.concatenate((next_state_deva, np.expand_dims(rgb_to_binary_mask(goal_bbox, [255, 255, 255]), axis=-1)), axis=2)
    next_state = torch.from_numpy(cv2.resize(next_state.astype(np.float32), (64, 64)).transpose(2, 0, 1)).float().cuda().unsqueeze(0)
    next_state, ht, ct = agent.CNN_LSTM.inference(next_state.unsqueeze(0), ht, ct)
    action = agent.get_action(next_state, eval=True)
    action=np.array(action)[0]
    assert len(action.shape)==2
    action = de_normalize(action)
    return action,ht,ct

def evaluate(env, agent, config, deva_model, deva_cfg, gd_model, sam_model):
    exp_manager = ExperienceManager(config.env,'/home/wuk/Instruction_Aware_Tracking/Tracking-Anything-with-DEVA/')
    # result_saver, deva = DEVATracker(deva_model, deva_cfg)
    recovery_state = RecoveryState()

    next_state = env.reset()
    env.unwrapped.unrealcv.set_max_speed(env.unwrapped.player_list[env.unwrapped.target_id], 80)
    env.unwrapped.unrealcv.set_appearance(env.unwrapped.player_list[env.unwrapped.tracker_id], 21)
    if 'Map_ChemicalPlant_1' not in config.env:
        env.unwrapped.unrealcv.set_cam(env.unwrapped.player_list[env.unwrapped.tracker_id],
                                       [40, 0, 0],
                                       [0, 0, 0])
    env.unwrapped.unrealcv.set_obj_scale(env.unwrapped.player_list[env.unwrapped.tracker_id], (0.5, 0.5, 0.5))
    done = False
    rewards=0
    eval_step=0
    ht=None
    ct=None
    recovery_success_cnt=0
    recovery_failure_cnt=0

    while not done:
        # Track object with DEVA
        # next_state_deva = DEVAProcess(next_state,deva,sam_model,gd_model,result_saver,deva_step)
        frame = cv2.hconcat((cv2.resize(next_state[0][:,:,0:3], (480, 480)), cv2.resize(next_state[0][:,:,3:], (480, 480))))
        # cv2.imshow('goal',bbox_goal[0])
        if len(recovery_state.failure_imgs)>0:
            frame_fail=cv2.hconcat(list(recovery_state.failure_imgs))
            cv2.imshow('fail images',frame_fail)
        if len(recovery_state.recovery_images)>0:
            frame_recover=cv2.hconcat(list(recovery_state.recovery_images))
            cv2.imshow('recovery images',frame_recover)
        cv2.imwrite('/home/wuk/Instruction_Aware_Tracking/Failure_recovery_demo/{}.png'.format(eval_step),frame)

        cv2.imshow('next_state', frame)
        cv2.waitKey(1)
        next_state_deva=next_state[0][:,:,3:]
        current_bbox = get_bounding_box(next_state_deva) # check if the target is still visible in the image
        if eval_step%8 ==0:
            recovery_state.failure_imgs.append(next_state[0][:,:,0:3])
        # Get action from tracking agent
        action,ht,ct = AgentPolicy(agent,next_state_deva, bbox_goal,ht,ct)
        # Handle tracking failure
        if eval_step>0 and current_bbox is None:#info['metrics']['target_viewed']==0
            if eval_step<10:#初始化失败直接退出
                break
            cmd = f'vset /action/game/pause'
            env.unwrapped.unrealcv.client.request(cmd)
            if not recovery_state.is_recovering:
                recovery_state.start_recovery()
                recovery_state.recovery_images.append(recovery_state.failure_imgs[-1])
                response=exp_manager.call_api(recovery_state)
                recovery_analysis, recovery_action=parse_recovery_analysis_actions(response)
                # recovery_analysis, recovery_action=exp_manager.call_api_v2(recovery_state)
                # print("Context analysis",recovery_analysis)
                # print("recover action:",recovery_action)
                # recovery_action= parse_recovery_action_v2(recovery_action)
                recovery_state.set_actions(recovery_action)

            # Execute recovery action
            if recovery_state.has_pending_actions():
                obs = next_state[0][:,:,0:3] #record images when executing recovery actions
                action = recovery_state.get_next_action(obs)
                action=[action]
                print('recovery action:',action)
            cmd = f'vset /action/game/resume'
            env.unwrapped.unrealcv.client.request(cmd)
        # Handle tracking recovery (current_bbox is not None -> recovery success)
        elif current_bbox is not None and recovery_state.is_recovering:
            # import pdb
            # pdb.set_trace()
            # cv2.imwrite('/home/wuk/Instruction_Aware_Tracking/bbox_goal/success.png', concatenated_image)

            exp_manager.create_experience(recovery_state.trajectory_actions,recovery_analysis, recovery_state.history_actions, None,None,True)
            recovery_state.complete_recovery(success=True)
            print('Step {}:Complete recovery !'.format(eval_step))
            recovery_success_cnt+=1



        # Environment update
        if 'jump' not in action:
            recovery_state.trajectory_actions.append(action)
            next_state, reward, done, info = env.step(action)
        else:
            env.unwrapped.unrealcv.set_jump(env.unwrapped.player_list[env.unwrapped.tracker_id])
            next_state, reward, done, info = env.step([[0, 0]])

        if recovery_state.has_pending_actions():
            time.sleep(1)
        rewards+=reward
        # Update metrics
        eval_step+=1

    cv2.destroyAllWindows()
    # Handle end of episode (recover fail)
    if recovery_state.is_recovering:
        fail_reason,suggestion=exp_manager.critic( recovery_state.recovery_images, recovery_analysis, recovery_state.history_actions) #picture, analysis, actions
        if fail_reason!='parse error' and suggestion!='parse error':
            exp_manager.create_experience(recovery_state.trajectory_actions,recovery_analysis, recovery_state.history_actions, fail_reason, suggestion, False)
        # exp_manager.save_experience_pool()
        recovery_state.complete_recovery(success=False)
        recovery_failure_cnt+=1
    return rewards, eval_step,recovery_success_cnt,recovery_failure_cnt

def eval_average(config, agent ,deva_model, deva_cfg,gd_model, sam_model):
    # time_dilate = 10
    early_d = 50
    env = gym.make(config.env)
    # if int(config.time_dilation) > 0:  # -1 means no time_dilation
    # env = time_dilation.TimeDilationWrapper(env, time_dilate)
    # if int(config.early_done) > 0:  # -1 means no early_done
    env = early_done.EarlyDoneWrapper(env,early_d)


    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env = configUE.ConfigUEWrapper(env, offscreen=True)
    env = agents.NavAgents(env, mask_agent=False)

    env.seed(random.randint(0,65533))
    print('start evaluate...')
    AR = []
    EL = []
    recovery_success_cnts=0
    recovery_failure_cnts=0
    start_time=time.time()
    while len(EL)<50:
        # try:
        reward, eval_steps,recovery_success_cnt,recovery_failure_cnt= evaluate(env, agent, config ,deva_model, deva_cfg,gd_model, sam_model)
        print('episode：',len(EL),'reward:', reward[0], ' el:', eval_steps)
        if eval_steps > 50:
        #     print('stop check bad case')
            recovery_success_cnts+=recovery_success_cnt
            recovery_failure_cnts+=recovery_failure_cnt
            AR.append(reward[0])
            EL.append(eval_steps)
        EL_tmp = np.array(EL)
        print("success rate:{}".format(np.array([EL_tmp == 500]).sum()))
        # except:
        #     print('some thing wrong')
        #     i=i-1
        # print('eval time: ', time.time()-start_time)

    AR_mean = sum(AR) / len(AR)
    AR_max = max(AR)
    AR_min = min(AR)
    EL_mean = sum(EL) / len(EL)
    EL_max = max(EL)
    EL_min = min(EL)
    print("AR：{},{},{}".format(AR_mean, AR_max - AR_mean, AR_min - AR_mean),"EL：{},{},{}".format(EL_mean, EL_max - EL_mean, EL_min - EL_mean))
    # print("EL：{},{},{}".format(EL_mean, EL_max - EL_mean, EL_min - EL_mean))
    EL_tmp = np.array(EL)
    print("success rate:{}".format(np.array([EL_tmp==500]).sum()))
    print("Success recovery cnt:",recovery_success_cnts)
    print("Failure recovery cnt:",recovery_failure_cnts)
    env.close()
    return AR_mean, EL_mean
def train(config ,deva_model, deva_cfg,gd_model, sam_model):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device,
                          lstm_seq_len=config.lstm_seq_len,config=config)
    buffer_path = config.buffer_path
    if 'train' in config.mode:
        buffer = load_Buffer(buffer, buffer_path, config)

    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0


    agent = CQLSAC_CNN_LSTM_GoalCondition_bbox(state_size=(4, 64, 64),
                            action_size=2,
                            tau=config.tau,
                            hidden_size=config.hidden_size,
                            learning_rate=config.learning_rate,
                            temp=config.temperature,
                            with_lagrange=config.with_lagrange,
                            cql_weight=config.cql_weight,
                            target_action_gap=config.target_action_gap,
                            device=device,
                            stack_frames=1,
                            lstm_seq_len=config.lstm_seq_len,
                            lstm_layer=config.lstm_layer,
                            lstm_out=config.lstm_out)



    if config.load_agent_model is not None:
        agent.load_state_dict(torch.load(os.path.join(config.load_agent_model)))
    AR, EL = eval_average(config, agent, deva_model, deva_cfg, gd_model, sam_model)


if __name__ == "__main__":
    config,deva_model, deva_cfg,gd_model, sam_model = get_config()
    train(config ,deva_model, deva_cfg,gd_model, sam_model)
