import random
from datetime import datetime
import re
import os
import json

# from moviepy.config import success
from openai import OpenAI
import cv2
from system_prompt import encode_image_array
from sklearn.preprocessing import normalize
from fastdtw import fastdtw  # DTW library for efficient DTW calculation
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
class ExperienceManager:
    def __init__(self, exp_name,experience_pool_file):
            self.exp_name=exp_name
            self.experience_pool_file = experience_pool_file
            self.experience_pool = []
            self.load_experience_pool()
    def load_experience_pool(self):
        """加载历史经验池"""
        map_name = self.exp_name.split('-ContinuousColorMask')[0].split('UnrealTrack-')[1]
        self.experience_pool_file=os.path.join(self.experience_pool_file,'VLM_Memory_'+map_name+'.json')
        if os.path.exists(self.experience_pool_file):
            with open(self.experience_pool_file, "r", encoding="utf-8") as f:
                self.experience_pool = json.load(f)
                self.experience_pool=[exp for exp in self.experience_pool if exp.get('id').startswith(self.exp_name)]
        else:
            self.experience_pool = []
            print(f"{self.experience_pool_file} does not exist. Please add previous experiences.")


    def save_experience_pool(self):
        """保存对经验池的更改"""
        with open(self.experience_pool_file, "w", encoding="utf-8") as f:
            json.dump(self.experience_pool, f, ensure_ascii=False, indent=4)



    def undate_experience_pool(self):
        # 参考后根据结果成功失败，修改被参考经验的得分
        # 多次失败的低分经验（比如得分低于0.2）可以删除
        return

    #Case部分，可以直接在生成的时候要他参考，这种情况suggestion更针对通用的，比如为了什么什么任务需要怎么做
    # 也可以把调用分成两次，第一次正常，第二次做一个revise，就是把第一次生成的目标和行动用来搜索经验，然后让VLM参考cases之后判断需不需要调整一下action序列以避免一些错误
    def call_api(self,recover_state):
        client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        # sk-xxx替换为自己的key
        api_key=''
        )

        # 假设设置2条高相似度高评分的经验和1条欠探索的新经验

        # trajectory_actions = list(recover_state.trajectory_actions)
        # Exp1,Exp2=self.get_exp(trajectory_actions)
        # Exp = [
        #     [Exp1.get("analysis"), Exp1.get("action"), Exp1.get("fail_reason"), Exp1.get("suggestion")],
        #     [Exp2.get("analysis"), Exp2.get("action"), Exp2.get("fail_reason"), Exp2.get("suggestion")]
        # ]


       #  prompt_reflection = f"""
       #  # Task
       #  You are an intelligent assistant.
       #  Your task is to help a visual tracking robot recover from losing the target person from view or being hindered by ground obstacles.
       #  Based on the robot's three continuous first-person view observations, which is sampled before failure with a 5-step interval.
       #  You need resoning a list of five continuous actions to help the robot recover from the failure, each action will be executed for one step.
       #  \n
       #  # The available actions are:
       #  - Move Forward: Propel the agent forward by 1 meter.
       #  - Move Backward: Propel the agent backward by 1 meter.
       #  - Turn Left: Turn left by 30 degrees.
       #  - Turn Right: Turn right by 30 degrees.
       #  - Jump Over: Leap over an obstacle directly in front of the robot if the obstacle is small and within jumping range, such as stairs, a green belt, or a box.
       #  \n
       #  #Case
       #  Please refer to these similar cases, containing both success and failure cases to complete your task and avoid the same mistakes.
       #
       #  Success Case
       #  Contxt analysis: {Exp[0][0]} Action: {Exp[0][1]}
       #
       #  Failure Case
       #  Contxt analysis: {Exp[1][0]} Action: {Exp[1][1]} Failure Reason: {Exp[1][2]} Suggestion: {Exp[1][3]}
       #  \n
       #  # Hint
       #  1.When the robot finish bypassing the hinder structure, it should turn to face the occluded area behind the hinder structure.
       #  2.When the target move out of view without occlusion effect, the robot should turn to the target's last known direction without moving forward.
       #
       #  # Output Format
       #  Your response should contain two elements: a context analysis and an action list that could help the agent recover from the failure. The final response should strictly follow this format:
       #  [Context analysis]:Based on the continuous sequence of observations, thinkg step-by-step: if the target occluded or hindered by object or surrounding structures? If yes, what object or structure is currently occluding the target? Describe the occluding object or structure and its spatial position relative to the robot.
       #  If no occlusion is present, the target has likely moved out of the robot's view, analyzing the last known position relative to the robot. Reasoning a list of actions to bring the target back into view.
       # [Recovery action]: [action 1, action 2, action 3, action 4, action 5]
       #  """

        # Exp1, Exp2,Exp3 = self.get_exp(trajectory_actions)
        # Exp = [
        #     [Exp1.get("analysis"), Exp1.get("action"), Exp1.get("fail_reason"), Exp1.get("suggestion")],
        #     [Exp2.get("analysis"), Exp2.get("action"), Exp2.get("fail_reason"), Exp2.get("suggestion")],
        #     [Exp3.get("analysis"), Exp3.get("action"), Exp3.get("fail_reason"), Exp3.get("suggestion")]
        # ]


        # prompt_reflection = f"""
        #        # Task
        #        You are an intelligent assistant.
        #        Your task is to help a visual tracking robot recover from losing the target person from view or being hindered by ground obstacles.
        #        Based on the robot's three continuous first-person view observations, which is sampled before failure with a 5-step interval.
        #        You need resoning a list of five continuous actions to help the robot recover from the failure, each action will be executed for one step.
        #        \n
        #        # The available actions are:
        #        - Move Forward: Propel the agent forward by 1 meter.
        #        - Move Backward: Propel the agent backward by 1 meter.
        #        - Turn Left: Turn left by 30 degrees.
        #        - Turn Right: Turn right by 30 degrees.
        #        - Jump Over: Leap over an obstacle directly in front of the robot if the obstacle is small and within jumping range, such as stairs, a green belt, or a box.
        #        \n
        #        #Case
        #        Please refer to these similar cases to complete your task and avoid the same mistakes.
        #        #Case1
        #        Contxt analysis: {Exp[0][0]} Action: {Exp[0][1]} Failure Reason: {Exp[0][2]} Suggestion: {Exp[0][3]}
        #
        #        \n
        #        # Hint
        #        1.When the robot finish bypassing the hinder structure, it should turn to face the occluded area behind the hinder structure.
        #        2.When the target move out of view without occlusion effect, the robot should turn to the target's last known direction without moving forward.
        #
        #        # Output Format
        #        Your response should contain two elements: a context analysis and an action list that could help the agent recover from the failure. The final response should strictly follow this format:
        #        [Context analysis]:Based on the continuous sequence of observations, thinkg step-by-step: if the target occluded or hindered by object or surrounding structures? If yes, what object or structure is currently occluding the target? Describe the occluding object or structure and its spatial position relative to the robot.
        #        If no occlusion is present, the target has likely moved out of the robot's view, analyzing the last known position relative to the robot. Reasoning a list of actions to bring the target back into view.
        #        [Recovery action]: [action 1, action 2, action 3, action 4, action 5]
        #        """


        prompt = f"""
        # Task
        You are an intelligent assistant.
        Your task is to help a visual tracking robot recover from losing the target person from view or being hindered by ground obstacles.
        Based on the robot's three continuous first-person view observations, which is sampled before failure with a 5-step interval, you need resoning a list of five continuous actions to help the robot recover from the failure, each action will be executed for one step.
        \n
        # The available actions are:
        - Move Forward: Propel the agent forward by 1 meter.
        - Move Backward: Propel the agent backward by 1 meter.
        - Turn Left: Turn left by 40 degrees.
        - Turn Right: Turn right by 40 degrees.
        - Jump Over: Leap over an obstacle directly in front of the robot if the obstacle is small and within jumping range, such as stairs, a green belt, or a box.
        \n
        # Hint
        1.When the robot finish bypassing the hinder structure, it should turn to face the occluded area behind the hinder structure.
        2.When the target move out of view without occlusion effect, the robot should turn to the target's last known direction without moving forward.
        \n
        # Output Format
        Your response should contain two elements: a context analysis and an action list that could help the agent recover from the failure. The final response should strictly follow this format:
        [Context analysis]:Based on the continuous sequence of observations, thinkg step-by-step: if the target occluded or hindered by object or surrounding structures? If yes, what object or structure is currently occluding the target? Describe the occluding object or structure and its spatial position relative to the robot.
        If no occlusion is present, the target has likely moved out of the robot's view, analyzing the last known position relative to the robot. Reasoning a list of actions to bring the target back into view.
        [Recovery action]: [action 1, action 2, action 3, action 4, action 5]
        """

        image_list=list(recover_state.failure_imgs)
        concatenated_image = cv2.hconcat(image_list)
        if len(image_list) > 0:
            cv2.imwrite('/home/wuk/Instruction_Aware_Tracking/bbox_goal/fail.png', concatenated_image)


        base64_image_list = [encode_image_array(image) for image in image_list]
        messages=[{"role": "system", "content": prompt}]
        for base64_image in base64_image_list:
            messages.append({"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    }
                }]
             })
        while True:
            try:
                response = client.chat.completions.create(
                    model='gpt-4o-2024-05-13',
                    max_tokens=300,
                    messages=messages,)
                break
            except:
                pass
        answer = response.choices[0].message.content

        analysis = answer.lower().split('recovery action')[0]
        if '[context analysis]:' in analysis:
            analysis = analysis.split('[context analysis]:')[1]
        elif 'context analysis' in analysis:
            analysis = analysis.split('context analysis')[1]
        else:
            analysis = analysis
        try:
            Exp1, Exp2, Exp3 = self.get_exp(analysis)
            Exp = [
                [Exp1.get("analysis"), Exp1.get("action"), Exp1.get("fail_reason"), Exp1.get("suggestion")],
                [Exp2.get("analysis"), Exp2.get("action"), Exp2.get("fail_reason"), Exp2.get("suggestion")],
                [Exp3.get("analysis"), Exp3.get("action"), Exp3.get("fail_reason"), Exp3.get("suggestion")]
            ]
        except:
            print("recall experience error, return None experience")
            Exp = [
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None]
            ]

        prompt_revise = f"""
                Considering your previous context analysis, please refer to these similar recovery cases to revise you recovery action sequence, avoiding the same mistakes.
                If the Failure Reason and Suggestion are None, then this case is success recovery case, otherwise the case is recovery failure case.
                #Case1
                Contxt analysis: {Exp[0][0]} Action: {Exp[0][1]} Failure Reason: {Exp[0][2]} Suggestion: {Exp[0][3]}
                #Case2
                Contxt analysis: {Exp[1][0]} Action: {Exp[1][1]} Failure Reason: {Exp[1][2]} Suggestion: {Exp[1][3]}
                #Case3
                Contxt analysis: {Exp[2][0]} Action: {Exp[2][1]} Failure Reason: {Exp[2][2]} Suggestion: {Exp[2][3]}
                The final output should still follow the format in the Output Format section, but the action list should be revised.

                Example output
                [Context analysis]: ...
                [Recovery action]: [action 1, action 2, action 3, action 4, action 5]
                """
        print(prompt_revise)
        messages.append({"role":"assistant","content":answer})
        messages.append({"role":"user","content":prompt_revise})
        while True:
            try:
                response = client.chat.completions.create(
                    model='gpt-4o-2024-05-13',
                    max_tokens=300,
                    messages=messages, )
                break
            except:
                pass
        answer = response.choices[0].message.content


        # image_list = [Image.fromarray(img) for img in image_list]
        # question = prompt
        # content = image_list + [question]
        # # msgs = [{'role': 'user', 'content': [image_list[-3],image_list[-2],image_list[-1], question]}]
        # msgs = [{'role': 'user', 'content': content}]
        # answer = model.chat(
        #     msgs=msgs,
        #     tokenizer=tokenizer
        # )


        return answer

    def call_api_v2(self,recover_state):
        client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        # sk-xxx替换为自己的key
        api_key=''
        )
        prompt_system = f"""
               # Task
               You are an intelligent assistant.
               Your task is to help a visual tracking robot recover from unknown failure cases.
               """
        prompt_failure_reason = f"""
        Based on the robot's three continuous first-person view observations, which is sampled before failure with a 5-step interval. 
        Formulate a context analysis by structuring the answer to the following tasks :
        Task 1:
        What caused the robot's lost the target person from view, choose from: <Occlusion by target's surrounding environment>, <Target's sudden turn caused out of view>, <Unknown Failure>.
        Task 2:
        If the target failed because Occlusion instead of target moving out of view, provide a short description of the structure or objects which caused failure, in the format: <Description of related Structure/Objects>, otherwise filled with "None". 
        Task 3:
        Based on the three observations, reasoning the possible current target position related to the robot, indicating the distance and direction related to robot, such as <Near the center with medium distance>, <Near the close right side>
        Your final response should contain the answer of the three task inside "<>", connected by "-".

        #Output format:
        Example 1
        <Target's sudden turn caused out of view> - <None> - <Close left side>
        """

        prompt_generate_action = f"""
        Generate five continuous actions to help the robot bring back the target into view.
        # All available actions:
        - Move Forward: Propel the agent forward by 2 meter.
        - Move Backward: Propel the agent backward by 2 meter.
        - Turn Left: Turn left by 40 degrees.
        - Turn Right: Turn right by 40 degrees.
        - Jump Over: Leap over an obstacle directly in front of the robot if the obstacle is small and within jumping range, such as stairs, a green belt, or a box.
        Your final response should be a list of five actions, such as:[Turn Left, Move Forward, Move Forward, Turn Right, Turn Right].
        """
        image_list = list(recover_state.failure_imgs)
        concatenated_image = cv2.hconcat(image_list)
        if len(image_list) > 0:
            cv2.imwrite('/home/wuk/Instruction_Aware_Tracking/bbox_goal/fail.png', concatenated_image)
        base64_image = encode_image_array(concatenated_image)
        messages = [{"role": "system", "content": prompt_system, },
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": prompt_failure_reason},
                         {
                             "type": "image_url",
                             "image_url": {
                                 "url": f"data:image/jpeg;base64,{base64_image}",
                             }
                         },
                     ],
                     }
                    ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,
        )
        failure_reason=response.choices[0].message.content
        messages.append({"role": "assistant", "content": failure_reason})
        messages.append({"role": "user", "content": prompt_generate_action})
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,
        )
        action_list = response.choices[0].message.content
        return failure_reason, action_list

    def critic(self,image_list, analysis, action):
        client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        # sk-xxx替换为自己的key
        api_key=''
        )

        prompt = f"""
        # BACKGROUND
        #Role: Robot Action Verification Expert
        You are responsible for comparing visual images and action sequences to reflect on 
        why the robot did not complete the movement task as he envisioned, and providing suggestions.

        # INPUT
        1. The thinking process of robot, which include their expected action outcomes: {analysis}
        2. Robot's action sequence: {action}
        3. First person perspective image sequence.

        # THINKING PROCESS
        1. analyze the image first to reflect their real behaviors
        2. Based on their actual action sequence, analyze why the execution of the actual action sequence leads to the behaviors in the picture instead of the expected result during the analysis
        3. Based on the differences analyzed in the second step, provide the reasons for the failure and make modification suggestions for the action sequence

        # OUTPUT
        <BEHAVIOR_ANALYSIS>
            <!-- Actual behavior reflected in the picture-->
            Example：After turning left once, the robot moved forward and then collided with a pillar
        </BEHAVIOR_ANALYSIS>

        <DISCREPANCY_ANALYSIS>
            <!-- Inconsistency between action sequence and target-->
            Example：The task wishes to bypass the left side the pillar to bring back the target peron into view, but it didn't adjust view to left after moving forward and exceeding the pillar. 
                The action trajectory shown in the picture indicates that the insufficient turn left after moving forward.
        </DISCREPANCY_ANALYSIS>

        <ADJUSTMENT_SUGGESTION>
            <!-- Provide actionable improvement suggestions.-->
            Example：Turn left should be made after move forward: [Turn right, Move Forward, Move Forward,Turn Left, Turn Left]
        </ADJUSTMENT_SUGGESTION>

        """
        prompt_reflection_v2 = f"""
        # BACKGROUND
        # Role: Robot Action Verification Expert
        You are responsible for analyzing why the robot did not complete the movement task as expected, based on the visual images, action sequences, and context analysis, and providing suggestions for improvement.

        # INPUT
        1. Context analysis from the robot's reasoning, including failure reasons in the format: <Failure reason> - <Description of related structure/objects> - <Last known target position>
        {analysis}
        2. Robot's actual action sequence:{action}
        3. First-person perspective image sequence after executing the recovery actions.
            
        # THINKING PROCESS
        1. Analyze the provided image sequence to reflect the robot’s actual behavior.
        2. Compare the actual action sequence to the expected outcome from the context analysis. Identify why the execution led to the observed behavior in the image instead of the expected result.
        3. Based on the identified discrepancies, provide the reasons for the failure and suggest modifications to the action sequence.
        
        # All available actions:
        - Move Forward: Propel the agent forward by 2 meter.
        - Move Backward: Propel the agent backward by 2 meter.
        - Turn Left: Turn left by 40 degrees.
        - Turn Right: Turn right by 40 degrees.
        - Jump Over: Leap over an obstacle directly in front of the robot if the obstacle is small and within jumping range, such as stairs, a green belt, or a box.
        
        # OUTPUT
        <BEHAVIOR_ANALYSIS>
            <!-- Actual behavior as seen in the image -->
            Example: After turning left once, the robot moved forward and collided with a pillar.
        </BEHAVIOR_ANALYSIS>

        <DISCREPANCY_ANALYSIS>
            <!-- Differences between the expected action and the actual behavior -->
            Example: The task required bypassing the pillar to the left to bring the target into view, but the robot did not adjust its view left after moving forward past the pillar. The image sequence shows the robot failed to turn left after moving forward.
        </DISCREPANCY_ANALYSIS>

        <ADJUSTMENT_SUGGESTION>
            <!-- Suggested changes to the action sequence -->
            Example: Add a turn left after moving forward: [Turn right, Move Forward, Move Forward, Turn Left, Turn Left].
        </ADJUSTMENT_SUGGESTION>
        """

        image_list=list(image_list)
        concatenated_image = cv2.hconcat(image_list)
        if len(image_list) > 0:
            cv2.imwrite('/home/wuk/Instruction_Aware_Tracking/bbox_goal/recovery.png', concatenated_image)

        base64_image = encode_image_array(concatenated_image)

        prompt = prompt.format(analysis=analysis, action=action)
        while True:
            try:
                response = client.chat.completions.create(
                    model='gpt-4o-2024-05-13',
                    max_tokens=500,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                }
                            }]
                        }
                    ], )
                break
            except:
                pass
        response = response.choices[0].message.content

        # image_list = [Image.fromarray(img) for img in image_list]
        # question = prompt
        # content = image_list + [question]
        # # msgs = [{'role': 'user', 'content': [image_list[-3],image_list[-2],image_list[-1], question]}]
        # msgs = [{'role': 'user', 'content': content}]
        # # t0 = time.time()
        # response = model.chat(
        #     msgs=msgs,
        #     tokenizer=tokenizer
        # )

        try:
            fail_reason = re.search(r"<DISCREPANCY_ANALYSIS>(.*?)</DISCREPANCY_ANALYSIS>", response, re.DOTALL).group(1).strip()
            suggestion = re.search(r"<ADJUSTMENT_SUGGESTION>(.*?)</ADJUSTMENT_SUGGESTION>", response, re.DOTALL).group(1).strip()
            print('reflection fail reason:',fail_reason)
            print('reflection suggestion:',suggestion)
        except:
            fail_reason = 'parse error'
            suggestion = 'parse error'
            print('reflection parse error:', response)

        return fail_reason,suggestion



    def create_experience(self,trajectory_actions, analysis, action, fail_reason, suggestion,success):
        new_exp = {
            "id": self.exp_name+'-'+datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "trajectory": list(trajectory_actions),
            "analysis": analysis,
            "action":action,
            "fail_reason": fail_reason,
            "suggestion": suggestion,
            "success_rate": success
        }

        # Example = {
        #     "id": "xxx",
        #     "claasification": "around the pillar", （如果图片embedding匹配相似度不好操作  可以考虑分类来匹配相似情况）
        #     "picture": PICTURE,
        #     "analysis": 人物在走到柱子后面后丢失了，我们现在应该追随人物，向左绕过柱子，尝试继续追踪,
        #     "action": [直行，左转，直行，直行，直行],
        #     "fail_reason": 智能体想要向左绕过柱子，但左转角度不够导致直行碰撞到柱子
        #     "suggestion": 直行后增加转向次数：[直行，左转，左转，左转，直行]
        #     "success_rate": 1.0
        # }

        self.experience_pool.append(new_exp)


    def get_exp(self,analysis):
        # 根据相似度、得分、或者探索等需求搜索经验 (自定义)

        #采取随机策略选取经验
        # Exp1,Exp2,Exp3 = random.choices(self.experience_pool, k=3)

        #对比经验池中的trajectory action 序列,找到失败前轨迹最相似的failure case. (DTW 方法)
        # Step 1: Compare the input trajectory with each trajectory in the experience pool
        # Success_similarities = []
        # Failure_similarities=[]
        # for exp in self.experience_pool:
        #     if 'trajectory' in exp.keys() and exp.get('success_rate'):
        #         experience_actions = exp.get('trajectory')  # Convert to action sequence
        #         # Step 2: Calculate the DTW distance between the input trajectory and the experience trajectory
        #         try:
        #             distance, _ = fastdtw(trajectory_actions, experience_actions)  # FastDTW is an efficient version of DTW
        #             Success_similarities.append((exp, distance))
        #         except:
        #             pass
        #     elif 'trajectory' in exp.keys() and not exp.get('success_rate'):
        #         experience_actions = exp.get('trajectory')  # Convert to action sequence
        #         try:
        #         # Step 2: Calculate the DTW distance between the input trajectory and the experience trajectory
        #             distance, _ = fastdtw(trajectory_actions, experience_actions)  # FastDTW is an efficient version of DTW
        #             Failure_similarities.append((exp, distance))
        #         except:
        #             pass
        # # Step 3: Sort experiences by similarity (lowest DTW distance)
        # Success_similarities.sort(key=lambda x: x[1])
        # Failure_similarities.sort(key=lambda x: x[1])
        #
        # Exp1 = Success_similarities[0][0]
        # Exp2 = Failure_similarities[0][0]

        # similarities = []
        # for exp in self.experience_pool:
        #     if 'trajectory' in exp.keys() and exp.get('success_rate'):
        #         experience_actions = exp.get('trajectory')
        #         distance, _ = fastdtw(trajectory_actions, experience_actions)  # FastDTW is an efficient version of DTW
        #         similarities.append((exp,distance))
        # similarities.sort(key=lambda x: x[1])
        # Exp1,Exp2,Exp3 = [similarities[i][0] for i in range(0,3)]


        vectorizer = TfidfVectorizer()
        similarities = []
        for exp in self.experience_pool:

            experience_context = exp.get('analysis')
            tfidf_matrix = vectorizer.fit_transform([experience_context,analysis])

            # Calculate cosine similarity between the two context analyses
            cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarities.append((exp,cos_sim[0][0]))
        similarities.sort(key=lambda x: x[1])
        Exp1,Exp2,Exp3 = [similarities[i][0] for i in range(0,3)]
        return Exp1,Exp2,Exp3




