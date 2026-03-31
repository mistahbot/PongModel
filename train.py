import ReplayBuffer
import NeuralNetwork
import logic
import numpy as np
import random

def normalize_state(game:logic.Game):
    state_dict = game.get_state()
    state = np.array([
        state_dict["pos_y_p1"] / 600,
        state_dict["pos_y_p2"] / 600,
        state_dict["pos_ball_x"] / 600,
        state_dict["pos_ball_y"] / 600,
        state_dict["velocity_ball_x"] / 4,
        state_dict["velocity_ball_y"] / 4
    ]).reshape(6, 1)
    return state

learning_rate = 0.0
discount_factor = 0.99
batch_size = 32
sync_interval = 20
epsilon = 1.0
epsilon_decay = 0.9999
epsilon_min = 0.05

env = logic.Game()
buffer = ReplayBuffer.ReplayBuffer(70000)

online_net = NeuralNetwork.NeuralNetwork(6, 64, 64, 3)
target_net = NeuralNetwork.NeuralNetwork(6, 64, 64, 3)
online_net.load_state()

target_net.copy_weights(online_net)


for episode in range (10000):
    env = logic.Game()
    state = normalize_state(env)
    done = False
    total_reward = 0

    while not done:
        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(online_net.Predict(state))

        """
        0 is stay
        1 is up
        2 is down
        """
        move_p2 = 0
        if action == 1:
            move_p2 = -1
        elif action == 2:
            move_p2 = 1

        d = env.get_state()
        move_p1 = 1 if d["pos_ball_y"] > d["pos_y_p1"] else -1
        if random.random() < 0.3:
            move_p1 = -move_p1

        reward, done = env.step(move_p1, move_p2)
        next_state = normalize_state(env)

        buffer.push(state, action, reward, next_state, done)

        if len(buffer.buffer) > batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            targets = []

            for i in range(batch_size):
                current_q = online_net.Predict(states[i]).copy()

                if dones[i]:
                    target_val = rewards[i]
                else:
                    target_val = rewards[i] + discount_factor * np.max(target_net.Predict(next_states[i]))
                
                current_q[actions[i], 0] = target_val
                targets.append(current_q)

            online_net.Train_Batch(states, targets, learning_rate)

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if episode % sync_interval == 0:
        target_net.copy_weights(online_net)
        print(f"Episode {episode} | Reward : {total_reward} | Epsilon {epsilon:.2f} | Score {env.get_score()}")

online_net.save_state()