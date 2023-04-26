import os
import sys
import numpy as np

import tensorflow as tf

from utils import get_data
from environment import RecEnv
from agents import DqnAgent, DdqnAgent
from constants import DATA_DIR, ALLOWED_ALGO, STATE_SIZE, SEED_VALUE


def evaluate_model(recommender, check_movies=False, top_k=False):
    # episodic reward
    episode_reward = 0
    steps = 0
    precision = 0
    correct_recommendation = 0
    # Environment
    user_id, items_ids, done = env.reset()
    if check_movies:
        print(f'user_id : {user_id}, rated_items_length:{len(env.user_items)}')
        print('history items : \n', np.array(env.get_items_names(items_ids)))

    while not done:
        # Find embeddings of user and items
        user_eb = recommender.embedding_network.get_layer('user_embedding')(np.array(user_id))
        items_eb = recommender.embedding_network.get_layer('movie_embedding')(np.array(items_ids))
        
        # Represent State
        state = recommender.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
        
        # Get most profitable action
        action = recommender.q.network(state)
        
        # Map action to recommended item
        recommended_item = recommender.recommend_item(action, env.recommended_items, top_k=top_k)
        if check_movies:
            print(f'recommended item id : {recommended_item}')
            print(f'recommened item : \n {np.array(env.get_items_names(recommended_item), dtype=object)}')
        
        # Calculate reward & observe new state (in env)
        next_items_ids, reward, done, _ = env.step(recommended_item)
        if reward > 0:
            correct_recommendation += 1
        reward = np.sum(reward)
        items_ids = next_items_ids
        episode_reward += reward
        steps += 1
    precision = correct_recommendation/steps
    if check_movies:
        print(f'precision : {precision}, episode_reward : {episode_reward}')
        print()
    
    return precision, episode_reward


if __name__ == "__main__":
    _,_, movies_list, ratings_df,_,movies_df = get_data()
    
    users_dict = np.load(os.path.join(DATA_DIR,'user_dict.npy'), allow_pickle=True)
    users_history_lens_dic = np.load(os.path.join(DATA_DIR,'users_histroy_lens_dict.npy'), allow_pickle=True)

    users_num = max(ratings_df["user_id"])+1
    items_num = max(ratings_df["movie_id"])+1

    # Evaluation setting
    eval_users_num = int(users_num * 0.2)
    eval_items_num = items_num
    eval_users_dict = {k:users_dict.item().get(k) for k in range(users_num-eval_users_num, users_num)}
    eval_users_history_lens = {k: users_history_lens_dic[k] for k in range(users_num-eval_users_num, users_num)}

    algorithm_choice = sys.argv[1]
    if algorithm_choice in ALLOWED_ALGO:
        print(f"Training: {ALLOWED_ALGO[algorithm_choice]}")
        np.random.seed(SEED_VALUE)
        tf.random.set_seed(SEED_VALUE)

        MAX_EPISODE_NUM = eval_users_num
        movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}

        env = RecEnv(eval_users_dict, eval_users_history_lens, movies_id_to_movies, STATE_SIZE)
        ACTION_SIZE = len(movies_df)
        if algorithm_choice == "DQN":
            recommender = DqnAgent(env, users_num, items_num, STATE_SIZE, ACTION_SIZE)
        else:
            recommender = DdqnAgent(env, users_num, items_num, STATE_SIZE, ACTION_SIZE)
        recommender.build_network()
        recommender.train(env, MAX_EPISODE_NUM, load_model=False)
    else:
        raise ValueError("Improper algorithm. Pass \"DQN\" or \"DDQN\"")
