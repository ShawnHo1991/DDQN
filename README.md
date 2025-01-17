# Double DQN with CartPole-v1 compare with DQN

Key Modifications:
Added target_model and periodically copied weights from model using set_weights.
Used model to choose actions (via np.argmax), but used target_model to calculate Q-values for the next state (np.argmax(model.predict(next_state)[0])).
Updated target_model every 10 episodes (update_target_frequency) to prevent Q-value overestimation and enhance learning stability.
With these changes, we have implemented Double DQN! 🚀

DDQN results: Average Reward: 9.50, Max Reward: 11.0, Min Reward: 8.0

Define replay

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

            # Training loop

Train DDQN

    episodes = 50  # More episodes to ensure sufficient training
    batch_size = 32  # Mini-batch size for replay training
    gamma = 0.95  # Discount factor for future rewards
    update_target_frequency = 10  # 每 10 回合更新一次 target model
     
    for e in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle tuple output
            state = state[0]
        #如果 state 是 tuple，則 只取 state[0] 作為狀態值。
        #這是為了適配新版 Gymnasium 的 env.reset()，因為舊版 Gym 只返回 state，而新版可能會返回 (state, info)。
    
        state = np.reshape(state, [1, state_size])
     
        for time in range(200):  # Max steps per episode
            # Choose action using epsilon-greedy policy
            action = act(state)
     
            # Perform action in the environment 不同版本的 Gym 和 Gymnasium 可能返回 4 個值或 5 個值
            result = env.step(action)
            if len(result) == 4:  # Handle 4-value output
                next_state, reward, done, _ = result
            else:  # Handle 5-value output
                next_state, reward, done, _, _ = result
     
            if isinstance(next_state, tuple):  # Handle tuple next_state
                next_state = next_state[0]
            next_state = np.reshape(next_state, [1, state_size])
     
            # Store experience in memory
            remember(state, action, reward, next_state, done)
     
            # Update state
            state = next_state
     
            if done:  # If episode ends
                print(f"Episode: {e+1}/{episodes}, Score: {time}, Epsilon: {epsilon:.2}")
                break
     
        # Train the model using replay memory
        if len(memory) > batch_size:
            replay(batch_size)
    
        # **每 N 回合更新一次 target model**
        if e % update_target_frequency == 0:
            target_model.set_weights(model.get_weights())
     
    env.close()
  
