import os
import numpy as np
import tensorflow as tf
import time

from core.q_learning import QN


class DQN(QN):
    """
    Abstract class for Deep Q Learning
    """

    def add_placeholders_op(self):
        raise NotImplementedError

    def get_q_values_op(self, scope, reuse=False):
        """
        set Q values, of shape = (batch_size, num_actions)
        """
        raise NotImplementedError

    def add_update_target_op(self, q_scope, target_q_scope):
        """
        Update_target_op will be called periodically 
        to copy Q network to target Q network
    
        Args:
            q_scope: name of the scope of variables for q
            target_q_scope: name of the scope of variables for the target
                network
        """
        raise NotImplementedError

    def add_copy_model_op(self, q, q_next):
        """
        :param q: the action value related to current state
        :param q_next: the action value related to next state
        :return:
        """
        raise NotImplementedError

    def init_well_trained_model(self):
        q_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q")
        t_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="well_trained_q")
        op = [tf.assign(t_param[i], q_param[i]) for i in range(len(q_param))]
        self.init_well_trained_model_op = tf.group(*op)

    # def add_copy_model_op(self, q_scope, target_q_scope):
    #     q_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
    #     t_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
    #     op = [tf.assign(t_param[i], q_param[i]) for i in range(len(q_param))]
    #     self.copy_model_op = tf.group(*op)

    def add_loss_op(self, q, target_q, next_q):
        """
        Set (Q_target - Q)^2
        """
        raise NotImplementedError

    def add_optimizer_op(self, scope):
        """
        Set training op wrt to loss for variable in scope
        """
        raise NotImplementedError

    def process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = tf.cast(state, tf.float32)
        state /= self.config.high

        return state

    def build(self):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        s = self.process_state(self.s)
        self.q = self.get_q_values_op(s, scope="q", reuse=False)

        # compute Q values of next state
        sp = self.process_state(self.sp)
        self.target_q = self.get_q_values_op(sp, scope="target_q", reuse=False)

        self.next_q = self.get_q_values_op(sp, scope="next_q", reuse=False)

        # add a well_trained model here, to help find the best action
        # self.well_trained_q = self.get_q_values_op(s, scope="well_trained_q", reuse=False)
        # self.init_well_trained_model()

        # share the parameters between q and next_q
        self.add_copy_model_op("q", "next_q")

        # add update operator for target network
        self.add_update_target_op("q", "target_q")

        # add square loss
        self.add_loss_op(self.q, self.target_q, self.next_q)

        # add optmizer for the main networks
        self.add_optimizer_op("q")

    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        self.sess = tf.Session()

        # tensorboard stuff
        self.add_summary()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # synchronise q and target_q networks
        self.sess.run(self.update_target_op)

        self.sess.run(self.copy_model_op)

        # self.sess.run(self.init_well_trained_model_op)

        # for saving networks weights
        self.saver = tf.train.Saver()
        # print(self.config.model_output)
        # print(tf.train.latest_checkpoint(self.config.model_output))
        self.saver.restore(self.sess, self.config.model_output)

        s = self.process_state(self.s)
        self.well_trained_q = self.get_q_values_op(s, scope="well_trained_q", reuse=False)
        self.init_well_trained_model()
        self.sess.run(self.init_well_trained_model_op)

        print("------------restore weights--------")

    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.succ_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="succ_rates")

        self.avg_q_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_q")
        self.max_q_placeholder = tf.placeholder(tf.float32, shape=(), name="max_q")
        self.std_q_placeholder = tf.placeholder(tf.float32, shape=(), name="std_q")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads_norm", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg_Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max_Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std_Reward", self.std_reward_placeholder)

        tf.summary.scalar("Avg_Q", self.avg_q_placeholder)
        tf.summary.scalar("Max_Q", self.max_q_placeholder)
        tf.summary.scalar("Std_Q", self.std_q_placeholder)

        tf.summary.scalar("succ_rates", self.succ_reward_placeholder)

        tf.summary.scalar("Eval_Reward", self.eval_reward_placeholder)

        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path,
                                                 self.sess.graph)

    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        self.saver.save(self.sess, self.config.model_output)

    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        # action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        # return np.argmax(action_values), action_values

        well_trained_values = self.sess.run(self.well_trained_q, feed_dict={self.s: [state]})[0]

        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        return np.argmax(well_trained_values), action_values

    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """
        # share the parameters between q and next_q
        # self.sess.run(self.copy_model_op)

        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size)

        fd = {
            # inputs
            self.s: s_batch,
            self.a: a_batch,
            self.r: r_batch,
            self.sp: sp_batch,
            self.done_mask: done_mask_batch,
            self.lr: lr,
            # extra info
            self.avg_reward_placeholder: self.avg_reward,
            self.max_reward_placeholder: self.max_reward,
            self.std_reward_placeholder: self.std_reward,
            self.avg_q_placeholder: self.avg_q,
            self.max_q_placeholder: self.max_q,
            self.std_q_placeholder: self.std_q,
            self.eval_reward_placeholder: self.eval_reward,
            self.succ_reward_placeholder: self.succ_rates,
        }

        # loss_eval, grad_norm_eval, summary, _, _, action, well_trained_q, q = self.sess.run([self.loss, self.grad_norm,
        #                                                                   self.merged, self.train_op,
        #                                                                   self.copy_model_op,self.a,self.well_trained_q,self.q],
        #                                                                  feed_dict=fd)

        loss_eval, grad_norm_eval, summary, _, action, well_trained_q, q = self.sess.run([self.loss, self.grad_norm,
                                                                                          self.merged, self.train_op,
                                                                                          self.a,
                                                                                          self.well_trained_q,
                                                                                          self.q],
                                                                                         feed_dict=fd)
        print("action", action)
        print("well_trained_q", well_trained_q)
        print("training q", q)

        # tensorboard stuff
        self.file_writer.add_summary(summary, t)

        return loss_eval, grad_norm_eval

    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.sess.run(self.update_target_op)

    def copy_model(self):
        self.sess.run(self.copy_model_op)
