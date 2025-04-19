import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time


Z_OFFSET = 0.05
XY_THRESHOLD = 0.01
Z_THRESHOLD = 0.005
GOAL_THRESHOLD = 0.02
MOVE_SPEED = 10.0
GRIPPER_DELAY = 0.5

gym.register_envs(gymnasium_robotics)
env = gym.make("FetchPickAndPlace-v4", render_mode="human")


def calculate_action(current_pos, target_pos, gripper_action):
    
    delta = (target_pos - current_pos) * MOVE_SPEED
    
    delta_clipped = np.clip(delta, -1.0, 1.0)
    return np.concatenate([delta_clipped, [gripper_action]])


observation_info = env.reset(seed=42)
observation = observation_info[0]

gripper_pos = observation['observation'][0:3]
object_pos = observation['observation'][3:6]
desired_goal = observation['desired_goal']

state = "MOVE_ABOVE_OBJECT"
last_gripper_action_time = -GRIPPER_DELAY

for step in range(5000):
    env.render()
    # time.sleep(0.01)

    gripper_pos = observation['observation'][0:3]
    object_pos = observation['observation'][3:6]
    
    object_rel_pos = observation['observation'][6:9]

    
    dist_gripper_obj_xy = np.linalg.norm(gripper_pos[:2] - object_pos[:2])
    dist_gripper_obj_z = np.abs(gripper_pos[2] - object_pos[2])
    dist_obj_goal = np.linalg.norm(object_pos - desired_goal)

    
    action = np.zeros(4)
    current_time = time.time()

    required_gripper_action = 1.0 # Open

    if state == "MOVE_ABOVE_OBJECT":
        target_pos = object_pos + np.array([0., 0., Z_OFFSET])
        action = calculate_action(gripper_pos, target_pos, 1.0) # Open
        if dist_gripper_obj_xy < XY_THRESHOLD and np.abs(gripper_pos[2] - target_pos[2]) < Z_THRESHOLD :
            print("State: MOVE_TO_OBJECT")
            state = "MOVE_TO_OBJECT"

    elif state == "MOVE_TO_OBJECT":
        target_pos = object_pos
        action = calculate_action(gripper_pos, target_pos, 1.0)
        
        if dist_gripper_obj_xy < XY_THRESHOLD and dist_gripper_obj_z < Z_THRESHOLD:
             if np.linalg.norm(object_rel_pos) < 0.015:
                print("State: GRASP")
                state = "GRASP"
                last_gripper_action_time = current_time

    elif state == "GRASP":
        action = calculate_action(gripper_pos, gripper_pos, -1.0) # Close gripper
        required_gripper_action = -1.0
        

        if current_time - last_gripper_action_time > GRIPPER_DELAY:
            print("State: LIFT_OBJECT")
            state = "LIFT_OBJECT"

    elif state == "LIFT_OBJECT":
        
        target_pos = gripper_pos + np.array([0., 0., Z_OFFSET])
        action = calculate_action(gripper_pos, target_pos, -1.0) # Keep closed
        required_gripper_action = -1.0
        

        if gripper_pos[2] > object_pos[2] + Z_OFFSET - 0.01: 
             print("State: MOVE_ABOVE_GOAL")
             state = "MOVE_ABOVE_GOAL"

    elif state == "MOVE_ABOVE_GOAL":
        target_pos = desired_goal + np.array([0., 0., Z_OFFSET])
        action = calculate_action(gripper_pos, target_pos, -1.0)
        required_gripper_action = -1.0
        

        dist_gripper_goal_xy = np.linalg.norm(gripper_pos[:2] - target_pos[:2])
        if dist_gripper_goal_xy < XY_THRESHOLD and np.abs(gripper_pos[2] - target_pos[2]) < Z_THRESHOLD:
            print("State: MOVE_TO_GOAL")
            state = "MOVE_TO_GOAL"

    elif state == "MOVE_TO_GOAL":
        target_pos = desired_goal
        action = calculate_action(gripper_pos, target_pos, -1.0) # Keep closed
        required_gripper_action = -1.0
        dist_gripper_goal_z = np.abs(gripper_pos[2] - target_pos[2])
        if dist_gripper_goal_z < Z_THRESHOLD :
             print("State: RELEASE")
             state = "RELEASE"
             last_gripper_action_time = current_time

    elif state == "RELEASE":
        action = calculate_action(gripper_pos, gripper_pos, 1.0) # Open gripper
        required_gripper_action = 1.0
        
        if current_time - last_gripper_action_time > GRIPPER_DELAY:
            print("State: DONE (or move away)")
            state = "DONE"

    elif state == "DONE":

        target_pos = gripper_pos + np.array([0., 0., Z_OFFSET * 2])
        action = calculate_action(gripper_pos, target_pos, 1.0) # Keep open
        required_gripper_action = 1.0
        
        
        if step > 10:
            pass

    
    action[3] = required_gripper_action

    observation_info = env.step(action)
    observation = observation_info[0]
    reward = observation_info[1]
    terminated = observation_info[2]
    truncated = observation_info[3]
    info = observation_info[4]

    if info.get('is_success', False):
        print("Goal Achieved!")
        time.sleep(1)
        terminated = True


    if terminated or truncated:
        print(f"Episode finished. State: {state}, Terminated: {terminated}, Truncated: {truncated}, Success: {info.get('is_success', False)}")
        observation_info = env.reset()
        observation = observation_info[0]
        state = "MOVE_ABOVE_OBJECT"
        last_gripper_action_time = -GRIPPER_DELAY
 
env.close()
print("Environment closed.")



