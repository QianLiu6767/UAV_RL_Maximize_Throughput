#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import trange
from sys import argv

fname_formats = ("ddrqn-dtc-{}.csv", "ddrqn-dtc-q-{}.csv")
results_path = "results/ddrqn-dtc/"
model_name = "ddrqn-model"
mode = True
if len(argv) > 1:
    if len(argv) > 2 and argv[2] == "watch":
        mode = False
    i = argv[1]
    model_name += '-' + i
    
    if mode:
        score_file = open(results_path + fname_formats[0].format(i), 'w+')
        q_file = open(results_path + fname_formats[1].format(i), 'w+')
else:
    score_file, q_file = \
            open(results_path + 'ddrqn-dtc.csv', 'w+'), \
            open(results_path + 'ddrqn-dtc-q.csv', 'w+')

# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 20
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 32

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 5

model_savefile = results_path + model_name + ".ckpt"
save_model = True and mode
load_model = False or not mode
skip_learning = False or not mode
# Configuration file path
#config_file_path = "../../scenarios/deathmatch.cfg"

config_file_path = "../../scenarios/defend_the_center.cfg"
#config_file_path = "../../scenarios/cig.cfg"
#config_file_path = "../../scenarios/rocket_basic.cfg"
#config_file_path = "../../scenarios/basic.cfg"

bots = 10

if config_file_path == "../../scenarios/basic.cfg" or config_file_path == "../../scenarios/rocket_basic.cfg" or config_file_path == "../../scenarios/simpler_basic.cfg":
    enemies_list = ['Cacodemon']
elif config_file_path == "../../scenarios/cig.cfg":
    '''
    enemies_list = ['Bond', 'Conan', 'T800', 'MacGyver', 'Plissken', 'Rambo', 'McClane', 'Machete', 
                    'Anderson', 'Leone', 'Predator', 'Ripley', 'Jones', 'Blazkowicz']
    '''
    enemies_list = ['CustomMarineRocket' for i in range(bots)]
else:
    enemies_list = ['HellKnight', 'Zombieman', 'Demon', 'ChaingunGuy', 'ShotgunGuy', 'MarineChainsawVzd']


num_enemies = len(enemies_list)


screen_format = 3
# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 3
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)
        #self.enemy = np.zeros(capacity, dtype=np.int32)
        enemy_shape = (capacity, num_enemies)
        self.enemy = np.zeros(enemy_shape, dtype=np.float32)

        q_shape = int((learning_steps_per_epoch/4)*epochs)
        self.q_index = 0
        self.q_values = np.zeros(q_shape, dtype=np.float32)
        self.actual_r = np.zeros(episodes_to_watch, dtype=np.float32)
        self.r_index = 0
        #self.x_pos = np.zeros(capacity, dtype=np.float32)
        #self.y_pos = np.zeros(capacity, dtype=np.float32)
        #self.z_pos = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward, enemy_present):
        self.s1[self.pos, :, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward
        self.enemy[self.pos, :] = enemy_present
        #self.x_pos[self.pos] = x
        #self.y_pos[self.pos] = y
        #self.z_pos[self.pos] = z

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        stop = self.size - sample_size
        j = randint(0, stop)
        end = j + sample_size
        i = np.array(range( j, end))
        #i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i], self.enemy[i]

    def add_q(self, q):
        self.q_values[self.q_index] = np.max(q)
        self.q_index = (self.q_index + 1) 

    def add_actual_r(self, r):
        self.actual_r[self.r_index] = r
        self.r_index = (self.r_index + 1)

    def get_q_history(self):
        return self.q_values, self.actual_r



def create_network(session, available_actions_count):
    # Create the input variables
    s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [screen_format], name="State")
    a_ = tf.placeholder(tf.int32, [None], name="Action")
    target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

    # Add 2 convolutional layers with ReLu activation
    conv1_1 = tf.contrib.layers.convolution2d(s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))

    conv1_2 = tf.contrib.layers.convolution2d(conv1_1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))

    conv1_2_flat = tf.contrib.layers.flatten(conv1_2)

    batch_size_var = tf.shape(conv1_2_flat)[0]

    lstm_size = batch_size
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    hidden_state = tf.zeros([batch_size_var, lstm_size])
    current_state = tf.zeros([batch_size_var, lstm_size])
    lstm_state = hidden_state, current_state
    # The value of state is updated after processing each batch.
    lstm_output, lstm_state = lstm(conv1_2_flat, lstm_state)

    q = tf.contrib.layers.fully_connected(lstm_output, num_outputs=available_actions_count, activation_fn=None,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))

    best_a = tf.argmax(q, 1)
    loss = tf.losses.mean_squared_error(q, target_q_)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    #########################################################################################################

    enemy_ = tf.placeholder(tf.float32, [None, num_enemies], name="enemy_")

    features_mlp = tf.contrib.layers.fully_connected(conv1_2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))


    enemy_ahead = tf.contrib.layers.fully_connected(features_mlp, num_outputs= num_enemies, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))

    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=enemy_, logits=enemy_ahead)
    loss_mlp = tf.reduce_mean(xentropy)
    train_step_mlp = optimizer.minimize(loss_mlp)

    with tf.name_scope("eval"):
        correct = tf.equal(tf.round(tf.nn.sigmoid(enemy_ahead)), tf.round(enemy_))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    
    #########################################################################################################
    
    def function_get_q_zeros():
            hidden_state = tf.zeros([batch_size, lstm_size])
            current_state = tf.zeros([batch_size, lstm_size])

    def function_learn_mlp(s1, enemy):
        feed_dict = {s1_: s1, enemy_: enemy}
        l, _ = session.run([loss_mlp, train_step_mlp], feed_dict=feed_dict)
        #acc_train = accuracy.eval(feed_dict=feed_dict, session=session)
        #print("Training enemy detection accuracy: ", acc_train)
        return l

    def function_learn(s1, target_q):
        feed_dict = {s1_: s1, target_q_: target_q}
        l, _ = session.run([loss, train_step], feed_dict=feed_dict)
        return l

    def function_get_q_values(state):
        return session.run(q, feed_dict={s1_: state})

    def function_get_best_action(state):
        return session.run(best_a, feed_dict={s1_: state})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, resolution[0], resolution[1], screen_format]))[0]

    return function_learn, function_get_q_values, function_simple_get_best_action, function_learn_mlp, function_get_q_zeros

def create_network2(session, available_actions_count):
    # Create the input variables
    s1_2_ = tf.placeholder(tf.float32, [None] + list(resolution) + [screen_format], name="State2")
    a_2_ = tf.placeholder(tf.int32, [None], name="Action")
    target_q_2 = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ2")

    conv2_1 = tf.contrib.layers.convolution2d(s1_2_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))

    conv2_2 = tf.contrib.layers.convolution2d(conv2_1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))

    conv2_2_flat = tf.contrib.layers.flatten(conv2_2)
    lstm_size = batch_size
    lstm2 = tf.contrib.rnn.LSTMCell(lstm_size)
    batch_size_var = tf.shape(conv2_2_flat)[0]

    hidden_state2 = tf.zeros([batch_size_var, lstm_size])
    current_state2 = tf.zeros([batch_size_var, lstm_size])
    lstm_state2 = hidden_state2, current_state2

    lstm_output2, lstm_state2 = lstm2(conv2_2_flat, lstm_state2)

    q2 = tf.contrib.layers.fully_connected(lstm_output2, num_outputs=available_actions_count, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))

    best_a2 = tf.argmax(q2, 1)
    loss2 = tf.losses.mean_squared_error(q2, target_q_2)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    train_step2 = optimizer.minimize(loss2)

    #########################################################################################################

    enemy_2 = tf.placeholder(tf.float32, [None, num_enemies], name="enemy_2")

    features_mlp2 = tf.contrib.layers.fully_connected(conv2_2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))


    enemy_ahead2 = tf.contrib.layers.fully_connected(features_mlp2, num_outputs= num_enemies, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))

    #xentropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=enemy_2,logits=enemy_ahead2)
    xentropy2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=enemy_2,logits=enemy_ahead2)
    loss_mlp2 = tf.reduce_mean(xentropy2)
    train_step_mlp2 = optimizer.minimize(loss_mlp2)

    with tf.name_scope("eval"):
        correct2 = tf.equal(tf.round(tf.nn.sigmoid(enemy_ahead2)), tf.round(enemy_2))
        accuracy2 = tf.reduce_mean(tf.cast(correct2, tf.float32))



    #########################################################################################################
    
    def function_get_q_zeros():
        hidden_state2 = tf.zeros([batch_size, lstm_size])
        current_state2 = tf.zeros([batch_size, lstm_size])

    def function_learn_mlp(s1, enemy):
        feed_dict = {s1_2_: s1, enemy_2: enemy}
        l, _ = session.run([loss_mlp2, train_step_mlp2], feed_dict=feed_dict)
        #acc_train = accuracy2.eval(feed_dict=feed_dict, session=session)
        #print("Training enemy detection accuracy: ", acc_train)
        return l

    def function_learn(s1, target_q):
        feed_dict = {s1_2_: s1, target_q_2: target_q}
        l, _ = session.run([loss2, train_step2], feed_dict=feed_dict)
        return l

    def function_get_q_values(state):
        return session.run(q2, feed_dict={s1_2_: state})

    def function_get_best_action(state):
        return session.run(best_a2, feed_dict={s1_2_: state})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, resolution[0], resolution[1], screen_format]))[0]

    return function_learn, function_get_q_values, function_simple_get_best_action, function_learn_mlp, function_get_q_zeros


def learn_from_memory(coin_flip):
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        sample_size = int(batch_size/4)
        s1, a, s2, isterminal, r, enemy = memory.get_sample(sample_size)
        get_q_zeros()
        get_q_zeros2()
        if coin_flip == 0:
            a_star = np.argmax(get_q_values(s2), axis=1)

            #a_star = np.max(get_q_values(s2), axis=1)#[0]
            #a_star = np.argmax(a_star)
            q2_b = get_q_values2(s2)
            memory.add_q(q2_b)
            q2_b_a_star = q2_b[np.arange(q2_b.shape[0]), a_star][4:sample_size]

            #q2_a_a = np.max(get_q_values(s1), axis=1)
            get_q_zeros()
            target_q = get_q_values(s1)[4:sample_size]

            target_q[np.arange(target_q.shape[0]), a[4:sample_size]] = r[4:sample_size] + (1 - isterminal[4:sample_size]) * (discount_factor * q2_b_a_star)

            get_q_zeros()
            learn(s1[4:sample_size], target_q)
            learn_mlp(s1, enemy)
            get_q_zeros()
        else:
            b_star = np.argmax(get_q_values2(s2), axis=1)

            q2_a = get_q_values(s2)
            memory.add_q(q2_a)
            q2_a_b_star = q2_a[np.arange(q2_a.shape[0]), b_star][4:sample_size]

            #q2_b_a = np.max(get_q_values(s1), axis=1)
            get_q_zeros2()
            target_q = get_q_values2(s1)[4:sample_size]
 
            target_q[np.arange(target_q.shape[0]), a[4:sample_size]] = r[4:sample_size] + (1 - isterminal[4:sample_size]) * (discount_factor * q2_a_b_star)

            get_q_zeros2()
            learn2(s1[4:sample_size], target_q)
            learn_mlp2(s1, enemy)
            get_q_zeros2()


def perform_learning_step(epoch, learning_step):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)
    coin_flip = randint(0,1)
    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        if coin_flip == 0:
            a = get_best_action(s1)
        else:
            a = get_best_action2(s1)
        #poss_a = np.array([get_q_values(s1),get_q_values2(s1)])
        #a = np.argmax(tf.reduce_mean([poss_a], 0))


    # Add variables for shaping the reward to be added to reward before memory.add_transition
    health_s1 = game.get_game_variable(GameVariable.HEALTH)
    ammo_s1 = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
    kills_s1 = game.get_game_variable(GameVariable.KILLCOUNT)

    # Enemy labeling
    state = game.get_state()
    labels = state.labels_buffer 
    #x, y, z = 0

    enemy_present = [0. for i in range(num_enemies)]
    
    for l in state.labels:
        #enemy_present = 1 if l.object_name in enemies_list else 0
        for i in range(num_enemies):
            if l.object_name == enemies_list[i]:
                enemy_present[i] = 1.

    enemy_present = np.array(enemy_present, dtype=np.float32)

    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None
    
    #health_shaping = game.get_game_variable(GameVariable.HEALTH) - health_s1 if not isterminal else 0
    #ammo_shaping = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO) - ammo_s1 if not isterminal else 0
    #kills_shaping = game.get_game_variable(GameVariable.KILLCOUNT) - kills_s1 if not isterminal else 0
    
    #reward += health_shaping + ammo_shaping + kills_shaping
    
    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward, enemy_present)

    if learning_step % 4 == 0:
    	learn_from_memory(coin_flip)
    


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    seed = 666
    # Sets the seed. It can be change after init.
    game.set_seed(seed)
    ########################################################################################
    if config_file_path == "../../scenarios/cig.cfg":
        game.set_doom_map("map01")  # Limited deathmatch.
    
        # game.set_doom_map("map02")  # Full deathmatch.

        # Start multiplayer game only with your AI (with options that will be used in the competition, details in cig_host example).
        game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                    "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
                    "+viz_respawn_delay 10 +viz_nocheat 1")

        # Name your agent and select color
        # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
        game.add_game_args("+name AI +colorset 0")
    ########################################################################################

    #game.set_console_enabled(True)
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of the current episode/level.
    game.set_automap_buffer_enabled(True)

    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
    game.set_render_hud(True)
    game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(True)
    game.set_render_weapon(True)
    #game.set_render_effects_sprites(False)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(True)
    game.set_window_visible(False)
    #game.set_depth_buffer_enabled(True)

    # Turns on the sound. (turned off by default)
    game.set_sound_enabled(False)

    # Sets the livin reward (for each move) to -1
    #game.set_living_reward(-1)
    game.set_mode(Mode.PLAYER)
    #game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.add_available_game_variable(GameVariable.KILLCOUNT)
    game.add_available_game_variable(GameVariable.DEATHCOUNT)
    #game.add_available_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
    game.init()
    print("Doom initialized.")
    return game


if __name__ == '__main__':
    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]


    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    session = tf.Session()

    learn, get_q_values, get_best_action, learn_mlp, get_q_zeros = create_network(session, len(actions))
    learn2, get_q_values2, get_best_action2, learn_mlp2, get_q_zeros2 = create_network2(session, len(actions))

    saver = tf.train.Saver()
    if load_model:
        print("Loading model from: ", model_savefile)
        saver.restore(session, model_savefile)
    else:
        init = tf.global_variables_initializer()
        session.run(init)
    print("Starting the training!")

    time_start = time()
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()
            for learning_step in trange(learning_steps_per_epoch, leave=False):
                perform_learning_step(epoch, learning_step)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    get_q_zeros()
                    get_q_zeros2()

                    #####################################
                    #if config_file_path == "../../scenarios/cig.cfg":
                        #game.send_game_command("removebots")
                        #for i in range(bots):
                            #game.send_game_command("addbot")
                    #####################################
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_episode = []
            test_scores = []
            running_kills = 0.0
            running_deaths = 0.0
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                get_q_zeros()
                get_q_zeros2()
                
                prev_deaths = game.get_game_variable(GameVariable.DEATHCOUNT)

                #####################################
                if config_file_path == "../../scenarios/cig.cfg":
                        game.send_game_command("removebots")
                        for i in range(bots):
                            game.send_game_command("addbot")
                #####################################
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    coin_flip = randint(0,1)
                    if coin_flip == 0:
                        best_action_index = get_best_action(state)
                    else:
                        best_action_index = get_best_action2(state)

                    game.make_action(actions[best_action_index], frame_repeat)
		
                kills = game.get_game_variable(GameVariable.KILLCOUNT)
                deaths = game.get_game_variable(GameVariable.DEATHCOUNT)
                ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
                score = game.get_total_reward()
                
                running_kills += kills
                died = deaths > prev_deaths
                running_deaths += 1 if died else 0
                print(unicode("{},{},{},{},{}".format(
                        kills, 
                        running_deaths, 
                        26 - ammo, 
                        running_kills/running_deaths, 
                        score)
                    ),
                    file=score_file, end=u'\n'
                )

                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            print(unicode(','.join([str(q) for q in memory.q_values])), file=q_file, end=u'\n')
            print("Saving the network weigths to:", model_savefile)
            saver.save(session, model_savefile)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()


    avg_num_states = []
    for _ in range(episodes_to_watch):
        num_states = 0
        i = 0
        game.new_episode()
        get_q_zeros()
        get_q_zeros2()
        #####################################
        #if config_file_path == "../../scenarios/cig.cfg":
            #game.send_game_command("removebots")
            #for i in range(bots):
                #game.send_game_command("addbot")
        #####################################
        
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            coin_flip = randint(0,1)
            if coin_flip == 0:
                best_action_index = get_best_action(state)
            else:
                best_action_index = get_best_action2(state)
            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

            num_states += 1

        avg_num_states.append(num_states)
        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
        memory.add_actual_r(score)

    avg_num_states = int(np.max(avg_num_states))
    q_values, actual_rewards = memory.get_q_history()
    
    real_reward = 26
    true_q_value = 0
    for i in range(avg_num_states):
        true_q_value += np.power(discount_factor, (avg_num_states-i)) / real_reward

    print(true_q_value)
##### q_values is history of q_values over training steps; save to file ############

if mode:
    score_file.close()
    q_file.close()
