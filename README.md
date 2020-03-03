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
