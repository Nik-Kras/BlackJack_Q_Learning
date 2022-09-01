from tf_agents.networks import network
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import q_policy
from tf_agents.specs import tensor_spec
import tensorflow as tf

env = suite_gym.load("Blackjack-v1")

# Example of a game play
env.action_space.seed(42)
observation = env.reset()
print("Initial observation: ", observation)
for _ in range(5):
    action = env.action_space.sample()
    time_step = env.step(action)
    print("****")
    print("Applied action: ", action)
    print("Current observation: ", time_step.observation)
    print("Current reward: ", time_step.reward)
    print("Done? ", time_step.step_type)
    if time_step.step_type == 2:
        observation = env.reset()
        print("New observation: ", observation)
env.close()
print("-----")
print("End of testing gameplay")

# *************************************************** TF ****************************************

# Environment validation with 5 epochs
utils.validate_py_environment(env, episodes=5)

# Wrapping the environment to TensorFlow
tf_env = tf_py_environment.TFPyEnvironment(env)

# Testing gameplay in TensorFlow wrap
print("Testing gameplay in TensorFlow wrap")
time_step = tf_env.reset()
num_steps = 3
transitions = []
reward = 0
for i in range(num_steps):
  action = tf.constant([i % 2])
  # applies the action and returns the new TimeStep.
  next_time_step = tf_env.step(action)
  transitions.append([time_step, action, next_time_step])
  reward += next_time_step.reward
  time_step = next_time_step
  print("&&&&")
  print("Applied action: ", action)
  print("Current observation: ", next_time_step.observation)
  print("Current reward: ", next_time_step.reward)
  print("Done? ", next_time_step.step_type)

np_transitions = tf.nest.map_structure(lambda x: x.numpy(), transitions)
print("Historical game trajectories: ")
print('\n'.join(map(str, np_transitions)))
print('Total reward:', reward.numpy)
print("-----")
print("End of testing TensorFlow gameplay")

# ****************************************** Q-Network ******************************************

print("___________________________________________________________________")
print("----------------------- Deep Q-Learning ---------------------------")
print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
print("")

print("observation_spec: ", tf_env.observation_spec())
print("time_step_spec: ", tf_env.time_step_spec())
print("action_spec: ", tf_env.action_spec())

input_tensor_spec = tf_env.observation_spec()
time_step_spec = tf_env.time_step_spec()
action_spec = tf_env.action_spec()

# input_tensor_spec = tensor_spec.TensorSpec((4,), tf.float32)
# time_step_spec = ts.time_step_spec(input_tensor_spec)
# action_spec = tensor_spec.BoundedTensorSpec((),
#                                             tf.int32,
#                                             minimum=0,
#                                             maximum=2)

# Create array of 3 <tf.Tensor: shape=(1,), dtype=int64, numpy=array([17], dtype=int64)> and put them to Q-Network

num_actions = action_spec.maximum - action_spec.minimum + 1

class QNetwork(network.Network):

  def __init__(self, input_tensor_spec, action_spec, num_actions=num_actions, name=None):
    super(QNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)
    self._sub_layers = [
        tf.keras.layers.Dense(num_actions),
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._sub_layers:
      inputs = layer(inputs)
    return inputs, network_state


# batch_size = 2
# observation = tf.ones([batch_size] + time_step_spec.observation.shape.as_list())
# time_steps = ts.restart(observation, batch_size=batch_size)

time_step1 = tf_env.reset()
time_step2 = tf_env.reset()
print("---- Real Time Step: ", time_step1)
time_steps = time_step1 # [time_step1, time_step2]
print("---- 2 batch Time Step: ", time_steps)

print(time_steps)

my_q_network = QNetwork(
    input_tensor_spec=input_tensor_spec,
    action_spec=action_spec)
my_q_policy = q_policy.QPolicy(
    time_step_spec, action_spec, q_network=my_q_network)
action_step = my_q_policy.action(time_steps)
distribution_step = my_q_policy.distribution(time_steps)

print('Action:')
print(action_step.action)

print('Action distribution:')
print(distribution_step.action)