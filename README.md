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
