# Double DQN with CartPole-v1 compare with DQN

DDQN results: Average Reward: 9.50, Max Reward: 11.0, Min Reward: 8.0


    def replay(batch_size):
        global epsilon
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # **Double DQN**
                best_action = np.argmax(model.predict(next_state)[0]) # 選擇最優動作 (由 model 決定)
                target = reward + gamma * target_model.predict(next_state)[0][best_action]  # 用 target_model 計算 Q 值
        
            target_f = model.predict(state)  # 取得當前 Q 值
            target_f[0][action] = target #target_f[0] 取出整個 Q 值陣列
            model.fit(state, target_f, epochs=1, verbose=0)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
  
