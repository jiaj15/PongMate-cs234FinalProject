# cs234 Final Project

ddl: 03/11(poster session)

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
