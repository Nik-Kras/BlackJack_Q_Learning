from tf_agents.networks import network
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import q_policy
from tf_agents.agents import DqnAgent
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import py_driver
from tf_agents.specs import tensor_spec

from tensorflow.python.keras import Sequential
import tensorflow as tf
import numpy as np

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
input_tensor_spec2 = [tf.TensorSpec(shape=(3,))]

print(" MY OWN  input_tensor_spec2: ", input_tensor_spec2)
my_time_step_spec = ts.time_step_spec(input_tensor_spec2) # My idea of rebuilding Time Step
time_step_spec = tf_env.time_step_spec()                  # Original time Step
action_spec = tf_env.action_spec()

print(" MY OWN time_step_spec: ", my_time_step_spec)

time_step = tf_env.reset()
print("---- Real Time Step: ", time_step)
# Convert observation tuple to observation tensor
my_observation = tf.convert_to_tensor(np.array([y[0] for y in time_step.observation]))
my_observation = tf.expand_dims(my_observation, axis=0)
print("----  MY OWN Observation: ", my_observation)

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
        # tf.keras.layers.Input((3,)),
        tf.keras.layers.Dense(num_actions),
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    print("called input: ", inputs)
    my_observation = tf.convert_to_tensor(np.array([y[0] for y in inputs]))
    my_observation = tf.expand_dims(my_observation, axis=0)
    print("my_observation: ", my_observation)
    inputs = tf.cast(my_observation, tf.float32)
    print("Final input: ", inputs)
    for layer in self._sub_layers:
      inputs = layer(inputs)
    print("Q-Values: ", inputs)
    return inputs, network_state

# batch_size = 2
# observation = tf.ones([batch_size] + time_step_spec.observation.shape.as_list())
# time_steps = ts.restart(observation, batch_size=batch_size)

##############################################################
my_q_network = QNetwork(
    input_tensor_spec=tf_env.observation_spec(),
    action_spec=tf_env.action_spec())

my_q_policy = q_policy.QPolicy(
    time_step_spec, action_spec, q_network=my_q_network)

time_step = ts.restart(time_step.observation)
print("Sending Time Step: ", time_step)

action_step = my_q_policy.action(time_step)
distribution_step = my_q_policy.distribution(time_step)

print('Action:')
print(action_step.action)

print('Action distribution:')
print(distribution_step.action)
##############################################################
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!!!!! Agent")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# Custom Q-Network ##############################################################
# fc_layer_params = (100, 50)
# action_tensor_spec = tf_env.action_spec()
# num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
#
# # Define a helper function to create Dense layers configured with the right
# # activation and kernel initializer.
# def dense_layer(num_units):
#     return tf.keras.layers.Dense(
#         num_units,
#         activation=tf.keras.activations.relu,
#         kernel_initializer=tf.keras.initializers.VarianceScaling(
#             scale=2.0, mode='fan_in', distribution='truncated_normal'))
#
# # QNetwork consists of a sequence of Dense layers followed by a dense layer
# # with `num_actions` units to generate one q_value per available action as
# # its output.
# dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
# q_values_layer = tf.keras.layers.Dense(
#     num_actions,
#     activation=None,
#     kernel_initializer=tf.keras.initializers.RandomUniform(
#         minval=-0.03, maxval=0.03),
#     bias_initializer=tf.keras.initializers.Constant(-0.2))
# q_net = Sequential(dense_layers + [q_values_layer])

# class QNetwork(network.Network):
#
#     def __init__(self, input_tensor_spec, action_spec, num_actions=num_actions, name=None):
#         super(QNetwork, self).__init__(
#             input_tensor_spec=input_tensor_spec,
#             state_spec=(),
#             name=name)
#
#         self._sub_layers = [
#             # tf.keras.layers.Input((3,)),
#             tf.keras.layers.Dense(num_actions),
#         ]
#
#     def call(self, inputs, step_type=None, network_state=()):
#         del step_type
#         for layer in self._sub_layers:
#           inputs = layer(inputs)
#         return inputs, network_state
#
# q_net = QNetwork(
#     input_tensor_spec = tf_env.action_spec(),
#     action_spec = tf_env.action_spec(),
#     num_actions = 2)

learning_rate = 1e-3
train_step_counter = tf.Variable(0)
agent = DqnAgent(
    time_step_spec = time_step_spec,
    action_spec = action_spec,
    q_network = my_q_network,
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)
agent.initialize()

##############################################################

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!!!!! Reply Buffer")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# Example
# data_spec =  (
#         tf.TensorSpec([3], tf.float32, 'action'),
#         (
#             tf.TensorSpec([5], tf.float32, 'lidar'),
#             tf.TensorSpec([3, 2], tf.float32, 'camera')
#         )
# )
# data_spec = agent.collect_data_spec
# batch_size = 2
# max_length = 100
# print("Agent expects input: ", agent.collect_data_spec)
#
# replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#     data_spec,
#     batch_size=batch_size,
#     max_length=max_length)
#
# # Testing writing to the buffer
#
# time_step = tf_env.reset()
# time_step_batched = (time_step, time_step)
#
# print("Time step: ", time_step)
# print("Time step batched: ", time_step_batched)
# replay_buffer.add_batch(time_step_batched)

##############################################################

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!!!!! Drivers")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# driver = py_driver.PyDriver(
#     env = tf_env,
#     policy = agent.collect_policy,
#     observers: Sequence[Callable[[trajectory.Trajectory], Any]],
#     transition_observers: Optional[Sequence[Callable[[trajectory.Transition], Any]]] = None,
#     max_steps: 4,
#     max_episodes: 20
# )




##############################################################
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!!!!! Training of the Agent")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# time_step = tf_env.reset()
# rewards = []
# steps = []
# num_episodes = 5
#
# for _ in range(num_episodes):
#   episode_reward = 0
#   episode_steps = 0
#   while not time_step.is_last():
#     action = tf.random.uniform([1], 0, 2, dtype=tf.int32)
#     time_step = tf_env.step(action)
#     episode_steps += 1
#     episode_reward += time_step.reward.numpy()
#   rewards.append(episode_reward)
#   steps.append(episode_steps)
#   time_step = tf_env.reset()
#
# num_steps = np.sum(steps)
# avg_length = np.mean(steps)
# avg_reward = np.mean(rewards)
#
# print('num_episodes:', num_episodes, 'num_steps:', num_steps)
# print('avg_length', avg_length, 'avg_reward:', avg_reward)
