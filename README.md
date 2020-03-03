# cs234 Final Project

ddl: 03/11(poster session)

Find action
```python
def isFire(action):
    """
    action: int
    env: Pong-v0
    return true if it is a FIRE action
    """
    ACTION_MEANINGS = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    action_mean = ACTION_MEANINGS[action]
    if action_mean.find('FIRE') == -1:
        return False
    else:
        return True
```

TODO

1. Try Pong-v0 env
* find default difficulty and modify
```python
print(env.ale.getAvailableDifficulties()) # print out AvailableDifficulties
env.ale.setDifficulty(1)
```
* change reward settings in Wrapper.py (Rewrite a wrapper class or change reward settings in MaxAndSkipEnv class)
* retrain

2. Test Tennis-v0 with stable-baselines


SYZ Recording

Make some changes to sort the action(fire and unfire)

- in deep_q_learning.py

```python
    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """

        well_trained_values = self.sess.run(self.well_trained_q, feed_dict={self.s: [state]})[0]

        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        return np.argmax(well_trained_values), action_values
```
- in q1_schedule.py

```python
    def get_action(self, best_action, q_values):
        """
        Returns a random action with prob epsilon, otherwise returns the best_action

        Args:
            best_action: int 
                best action according some policy
        Returns:
            an action
        """

        ##############################################################
        ############# CLASSIFY FIRE AND UNFIRE VERSION ###############

        def isFire(action):
            """
            action: int
            env: Pong-v0
            return true if it is a FIRE action
            """
            ACTION_MEANINGS = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
            action_mean = ACTION_MEANINGS[action]
            if action_mean.find('FIRE') == -1:
                return False
            else:
                return True

        if isFire(best_action):
            candidate_actions = np.argsort(q_values)[::-1][:]
            fire_actions = []
            for a in candidate_actions:
                if isFire(a):
                    fire_actions.append(a)
            fire_actions = np.array(fire_actions)
            if np.random.random() < self.epsilon:
                return np.random.choice(fire_actions, 1)
            else:
                return fire_actions[0]

        else:
            return best_action
```

- in deep_q_learning.py in build() method
```python
        # add a well_trained model here, to help find the best action
        self.well_trained_q = self.get_q_values_op(s, scope="well_trained_q", reuse=False)
        self.init_well_trained_model()   
```
```python 
    def init_well_trained_model(self):

        q_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q")
        t_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="well_trained_q")
        op = [tf.assign(t_param[i], q_param[i]) for i in range(len(q_param))]
        self.sess.run(tf.group(*op))
```
