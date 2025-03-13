import time

import torch
import numpy as np
# from deva import DEVAInferenceCore
# from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
# from deva.ext.grounding_dino import get_grounding_dino_model
# from deva.inference.eval_args import add_common_eval_args, get_model_and_config
# from deva.inference.result_utils import ResultSaver
# from deva.ext.with_text_processor import process_frame_with_text as process_frame
import cv2
from PIL import Image, ImageDraw, ImageFont
import cv2, json, textwrap, tempfile, torch, torchaudio

def de_normalize(action):
    min_val = np.array([-30, -100])  # need to be modified for different binary
    max_val = np.array([30, 100])
    # 将数据从-1到1的范围反向归一化到0到1的范围
    denormalized_data = (action + 1) / 2

    # 将数据从0到1的范围反向归一化到原始范围
    denormalized_data = denormalized_data * (max_val - min_val) + min_val
    # need to be modified for different binary

    action = [[denormalized_data[0][0], denormalized_data[0][1]]]
    return action
def generate_bbox_goal(image_size, center, bbox_height, bbox_width):
    # Create an empty image
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # Calculate the top left corner of the bbox
    top_left = (center[0] - bbox_width // 2, center[1] - bbox_height // 2)

    # Calculate the bottom right corner of the bbox
    bottom_right = (center[0] + bbox_width // 2, center[1] + bbox_height // 2)

    # Draw the bbox on the image
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1)
    cv2.imwrite('../bbox_goal/eval_bbox_goal.jpg', image)
    return image
def get_bounding_box(mask_image):
    """
    获取三维mask图像中目标区域的边界框

    :param mask_image: 三维mask图像（高度, 宽度, 3）
    :return: 边界框的坐标 (x_min, y_min, x_max, y_max)
    """
    # 找到目标区域 (255, 255, 255) 的所有位置
    target_pixels = np.where(np.all(mask_image == [255, 255, 255], axis=-1))

    if len(target_pixels[0]) == 0:
        return None  # 没有找到目标区域

    # 计算边界框的坐标
    y_min = np.min(target_pixels[0])
    y_max = np.max(target_pixels[0])
    x_min = np.min(target_pixels[1])
    x_max = np.max(target_pixels[1])

    return x_min, y_min, x_max, y_max
def draw_rectangular_bbox(image, normalized_bbox, color=(255, 255, 255), thickness=2):
    """
    在图像上绘制归一化的矩形边界框

    :param image: 输入图像
    :param normalized_bbox: 归一化的边界框坐标 [cx, cy, w, h]
    :param color: 边界框颜色
    :param thickness: 边界框线条粗细
    :return: 绘制了边界框的图像
    """
    image_height, image_width = image.shape[:2]
    cx, cy, w, h = normalized_bbox

    # Denormalize the coordinates
    x_min = int((cx - w / 2) * image_width)
    y_min = int((cy - h / 2) * image_height)
    x_max = int((cx + w / 2) * image_width)
    y_max = int((cy + h / 2) * image_height)

    # Draw the bounding box
    overlay = image.copy()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color,2)
    alpha = 0.5  # Transparency factor.
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0,image)
    return image

def normalize_bbox(target_bbox, image_size):
    """
    Normalize the bounding box coordinates to the range [0, 1].

    Parameters:
    - target_bbox: The target bounding box in the format [cx, cy, w, h].
    - image_size: The size of the image as a tuple (width, height).

    Returns:
    - A list containing the normalized bounding box [cx, cy, w, h].
    """
    image_width, image_height = image_size
    cx, cy, w, h = target_bbox

    # Normalize the coordinates
    normalized_cx = cx / image_width
    normalized_cy = cy / image_height
    normalized_w = w / image_width
    normalized_h = h / image_height

    return [normalized_cx, normalized_cy, normalized_w, normalized_h]
def generate_bbox_goal(image_size, center, bbox_height, bbox_width):
    # Create an empty image
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # Calculate the top left corner of the bbox
    top_left = (center[0] - bbox_width // 2, center[1] - bbox_height // 2)

    # Calculate the bottom right corner of the bbox
    bottom_right = (center[0] + bbox_width // 2, center[1] + bbox_height // 2)

    # Draw the bbox on the image
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1)
    cv2.imwrite('./bbox_goal/eval_bbox_goal.jpg', image)
    return image

def generate_new_bbox_image(normalized_bbox, image_shape, delta_changes):
    height, width = image_shape
    cx, cy, w, h = normalized_bbox
    delta_cx, delta_cy, delta_w, delta_h = delta_changes

    # Denormalize the coordinates
    cx_pixel = int((cx+delta_cx) * width)
    cy_pixel = int((cy+delta_cy) * height)
    w_pixel = int((w+delta_w) * width)
    h_pixel = int((h+delta_h) * height)
    x_min = int(cx_pixel - (w_pixel/2))
    x_max = int(x_min + w_pixel)
    y_min = int(cy_pixel - (h_pixel/2))
    y_max = int(y_min + h_pixel)
    # print('new_bbox:',x_min, y_min, x_max, y_max)


    # Ensure the bounding box stays within the image boundaries
    # x_min = np.clip(x_min, 0, width)
    # new_cy = np.clip(x_max, 0, height)
    # new_w = np.clip(new_w, 1, width - new_cx)
    # new_h = np.clip(new_h, 1, height - new_cy)

    # Convert center coordinates to corner coordinates
    # x_min = new_cx - new_w // 2
    # y_min = new_cy - new_h // 2
    # x_max = x_min + new_w
    # y_max = y_min + new_h

    # Create a blank image
    bbox_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw the new bounding box
    cv2.rectangle(bbox_image, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)

    return bbox_image,x_min,y_min,x_max,y_max
def binary_to_bbox_mask(binary_mask):
    """
    Convert a binary mask to a bounding box mask.

    :param binary_mask: Binary mask image (height, width)
    :return: Bounding box mask image (height, width)
    """
    # Find the bounding box coordinates
    target_pixels = np.where(binary_mask == 255)
    if len(target_pixels[0]) == 0:
        return np.zeros_like(binary_mask)  # Return an empty mask if no target found

    y_min = np.min(target_pixels[0])
    y_max = np.max(target_pixels[0])
    x_min = np.min(target_pixels[1])
    x_max = np.max(target_pixels[1])

    # Create a new mask with the bounding box drawn on it
    bbox_mask = np.zeros_like(binary_mask)
    bbox_mask[y_min:y_max+1, x_min:x_max+1] = 255

    return bbox_mask
def rgb_to_binary_mask(rgb_image, target_color):
    """
    将RGB mask图像转换为二进制mask图像

    :param rgb_image: RGB mask图像
    :param target_color: 目标颜色 (B, G, R)
    :return: 二进制mask图像
    """
    # 创建一个与输入图像大小相同的空白二进制图像
    binary_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

    # 找到目标颜色的所有位置
    target_pixels = np.where(np.all(rgb_image == target_color, axis=-1))

    # 将目标颜色的像素值设置为1
    binary_mask[target_pixels] = 255

    return binary_mask


def iou(state, goal):
    if state.max()==255:
        boxA = get_bounding_box(state)
        debug_state = state.copy()
        # cv2.rectangle(debug_state, (boxA[0], boxA[1]), (boxA[2], boxA[3]), (255, 255, 255), 2)
        boxB= get_bounding_box(goal)
        # cv2.rectangle(debug_state, (boxB[0], boxB[1]), (boxB[2], boxB[3]), (0, 0, 255), -1)

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        text = str(iou)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, 20)  # 文本位置
        fontScale = 0.5
        color = (255, 255, 255)  # 白色
        thickness = 1
        # cv2.putText(debug_state, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        # cv2.imshow('debug_state', debug_state)
        # cv2.waitKey(1)
    else:
        iou=0
        # return the intersection over union value
    return iou

def reward_cal(state, goal):
    boxA = get_bounding_box(state)
    debug_state = state.copy()
    if state.max()==255 and boxA is not None:

        x_mid_a = (boxA[0] + boxA[2]) / 2
        y_mid_a = (boxA[1] + boxA[3]) / 2
        debug_state = state.copy()
        cv2.rectangle(debug_state, (boxA[0], boxA[1]), (boxA[2], boxA[3]), (255, 255, 255), 2)
        boxB = get_bounding_box(goal)
        x_mid_b = (boxB[0] + boxB[2]) / 2
        y_mid_b = (boxB[1] + boxB[3]) / 2
        cv2.rectangle(debug_state, (boxB[0], boxB[1]), (boxB[2], boxB[3]), (0, 0, 255), 2)

        #distance between the center of the bounding box
        distance = np.sqrt((x_mid_a - x_mid_b) ** 2 + (y_mid_a - y_mid_b) ** 2)

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        goal_ratio = boxBArea / (state.shape[0] * state.shape[1])
        # goal_ratio = boxAArea/boxBArea

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # iou = interArea / float(boxBArea)

        # area ratio of two bounding box
        area_ratio = boxAArea / boxBArea

        # reward function
        distance_reward = 1 - (distance / state.shape[0])

        # 计算IOU奖励，IOU越大奖励越高
        iou_reward = iou

        # 计算面积比例奖励，比例越接近1奖励越高
        ratio_reward = 1 - abs(1 - area_ratio)

        # 合并各项奖励，可以根据具体任务对权重进行调整
        # total_reward = 0.3333* distance_reward + 0.3333* iou_reward + 0.3333 * ratio_reward
        # if iou_reward>0:
        #     total_reward = iou_reward
        # else:
        # total_reward = (1-iou_reward)*distance_reward * (0.005/goal_ratio)  +0.7*iou_reward
        # total_reward  = 0.3*distance_reward +0.7*iou_reward
        #     total_reward = distance_reward * (0.001/goal_ratio)
        total_reward = iou_reward
        # 确保奖励值在0到1之间
        total_reward = np.clip(total_reward, -1, 1)

        text = str(total_reward)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, 20)  # 文本位置
        fontScale = 0.5
        color = (255, 255, 255)  # 白色
        thickness = 1
        cv2.putText(debug_state, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('debug_state', debug_state)
        cv2.waitKey(50)
    else:
        total_reward=-1

    return total_reward, debug_state

def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
def collect_random_deva(env, dataset, num_samples=200, deva_model=None, deva_cfg=None, gd_model=None, sam_model=None):
    torch.autograd.set_grad_enabled(False)
    deva_cfg['temporal_setting'] = 'online'
    assert deva_cfg['temporal_setting'] in ['semionline', 'online', 'window']
    deva_cfg['enable_long_term_count_usage'] = True
    deva = DEVAInferenceCore(deva_model, config=deva_cfg)
    deva.next_voting_frame = deva_cfg['num_voting_frames'] - 1
    deva.enabled_long_id()
    result_saver = ResultSaver('./deva_out', None, dataset='demo', object_manager=deva.object_manager)

    state = env.reset()
    state_deva = process_frame(deva, gd_model, sam_model, str(0) + '.jpg', result_saver, 0,
                                    image_np=state[0][:, :, 0:3].astype(np.uint8))
    state = state_deva
    # state = torch.from_numpy(cv2.resize(state.astype(np.float32), (64, 64)).transpose(2, 0, 1)).float().cuda()
    state = cv2.resize(state.astype(np.float32), (64, 64)).transpose(2, 0, 1)

    for i in range(num_samples):
        action = env.action_space[0].sample()
        next_state, reward, done, _ = env.step([[action[0],action[1]]])
        next_state_deva = process_frame(deva, gd_model, sam_model, str(i+1) + '.jpg', result_saver, i+1,
                                        image_np=next_state[0][:, :, 0:3].astype(np.uint8))
        next_state = next_state_deva
        # next_state = torch.from_numpy(cv2.resize(next_state.astype(np.float32), (64, 64)).transpose(2, 0, 1)).float().cuda()
        next_state = cv2.resize(next_state.astype(np.float32), (64, 64)).transpose(2, 0, 1)

        dataset.add(state,
                    [[action[0],action[1]]], reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
            state_deva = process_frame(deva, gd_model, sam_model, str(i+1) + '.jpg', result_saver, i+1,
                                       image_np=state[0][:, :, 0:3].astype(np.uint8))
            state = state_deva
            # state = torch.from_numpy(cv2.resize(state.astype(np.float32), (64, 64)).transpose(2, 0, 1)).float().cuda()
            state = cv2.resize(state.astype(np.float32), (64, 64)).transpose(2, 0, 1)
def de_normalize(action):
    min_val = np.array([-30, -100])  # need to be modified for different binary
    max_val = np.array([30, 100])
    # 将数据从-1到1的范围反向归一化到0到1的范围
    denormalized_data = (action + 1) / 2

    # 将数据从0到1的范围反向归一化到原始范围
    denormalized_data = denormalized_data * (max_val - min_val) + min_val
    # need to be modified for different binary
    action = [[denormalized_data[0][0], denormalized_data[0][1]]]
    return action
def evaluate(env, policy, eval_runs=5):
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()

        rewards = 0
        while True:
            action = policy.get_action(state, eval=True)

            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)



def create_board(height, width, font_path, font_size, interval, initial_text='', user_logo = None, assistant_logo = None):
    image = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    max_chars = width // draw.textbbox((0, 0), 'A', font=font)[2]
    if initial_text:
        draw.text((0, interval), initial_text, fill=(255, 255, 255), font=font)
        last = np.array([0, interval + font_size])
    else:
        last = np.array([0, 0])
    board = {'image': image, 'draw': draw, 'max_chars': max_chars, 'font': font, 'font_size': font_size, 'last': last, 'width': width, 'height': height, 'interval': interval}
    if assistant_logo:
        assistant_logo_ratio = assistant_logo.width / assistant_logo.height
        assistant_logo_height = font_size
        assistant_logo_width = int(assistant_logo_height * assistant_logo_ratio)
        resized_assistant_logo = assistant_logo.resize((assistant_logo_width, assistant_logo_height))
        board['assistant_logo'] = resized_assistant_logo
    if user_logo:
        user_logo_ratio = user_logo.width / user_logo.height
        user_logo_height = font_size
        user_logo_width = int(user_logo_height * user_logo_ratio)
        resized_user_logo = user_logo.resize((user_logo_width, user_logo_height))
        board['user_logo'] = resized_user_logo
    return board

def append_text(board: dict, text: str, with_logo: str = None):
    wrapped_lines = textwrap.wrap(text, width=board['max_chars'], break_long_words=True, replace_whitespace=True)
    for i, line in enumerate(wrapped_lines):
        board['last'] += np.array([0, board['interval']])
        if with_logo and i == 0:
            board['image'].paste(board[with_logo], board['last'].tolist(), board[with_logo])
            board['draw'].text((board['last'][0] + board[with_logo].width, board['last'][1]), line, fill=(255, 255, 255), font=board['font'])
        else:
            board['draw'].text(board['last'], line, fill=(255, 255, 255), font=board['font'])
        board['last'] += np.array([0, board['font_size']])

def update_text(board: dict, text: str):
    board['draw'].rectangle([0, 0, board['width'], board['height']], fill=(0, 0, 0))
    board['last'] = np.array([0, 0])
    append_text(board, text)

def create_live_boards(height, user_logo, assistant_logo, font_path, width=500):
    state_board = create_board(height=30, width=width, font_path=font_path, font_size=20, interval=10, initial_text='Waiting...')
    conversation_board = create_board(user_logo=user_logo, assistant_logo=assistant_logo, height=height-30, width=width, font_path=font_path, font_size=15, interval=5, initial_text='')
    return state_board, conversation_board