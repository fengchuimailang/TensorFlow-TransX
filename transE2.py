# coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes

ll = ctypes.cdll.LoadLibrary  # ctypes是python的一个函数库，提供和C语言兼容的数据类型，可以直接调用动态链接库中的导出函数
lib = ll("./init.so")  # 初始化动态链接库
test_lib = ll("./test.so")  # 测试动态链接库


class Config(object):

    def __init__(self):
        path = "./data/FB15K/"
        lib.setInPath(path, len(path))  # 标准输出打印 init path... 函数无返回值
        test_lib.setInPath(path, len(path))  # 标准输出打印 init path...
        lib.setBernFlag(0)  # bernFlag = flag;
        self.learning_rate = 0.001
        self.testFlag = False  # 是否测试
        self.loadFromData = False  # 是否加载数据
        self.L1_flag = True  # 是否 l1 正则化
        self.hidden_size = 100
        self.nbatches = 100  # bitch size
        self.entity = 0  # 实体数量
        self.relation = 0  # 关系数量
        self.trainTimes = 1000  # 训练次数
        self.margin = 1.0  # 计算loss的margin


class TransEModel(object):  # 模型主体

    def __init__(self, config):

        entity_total = config.entity
        relation_total = config.relation
        batch_size = config.batch_size
        size = config.hidden_size  # 隐层神经元数量
        margin = config.margin

        # 正样本
        self.pos_h = tf.compat.v1.placeholder(tf.int32, [None])  # 头
        self.pos_t = tf.compat.v1.placeholder(tf.int32, [None])  # 尾
        self.pos_r = tf.compat.v1.placeholder(tf.int32, [None])  # 关系

        # 负样本
        self.neg_h = tf.compat.v1.placeholder(tf.int32, [None]) # 头
        self.neg_t = tf.compat.v1.placeholder(tf.int32, [None]) # 尾
        self.neg_r = tf.compat.v1.placeholder(tf.int32, [None]) # 关系

        with tf.compat.v1.name_scope("embedding"): # 操作名称 embedding
            self.ent_embeddings = tf.compat.v1.get_variable(name="ent_embedding", shape=[entity_total, size],
                                                            initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                                                scale=1.0, mode="fan_avg", distribution=(
                                                                    "uniform" if False else "truncated_normal")))
            self.rel_embeddings = tf.compat.v1.get_variable(name="rel_embedding", shape=[relation_total, size],
                                                            initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                                                scale=1.0, mode="fan_avg", distribution=(
                                                                    "uniform" if False else "truncated_normal"))) # truncated_normal 两标准差以内的正态分布
            pos_h_e = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.pos_t)
            pos_r_e = tf.nn.embedding_lookup(params=self.rel_embeddings, ids=self.pos_r)
            neg_h_e = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(params=self.ent_embeddings, ids=self.neg_t)
            neg_r_e = tf.nn.embedding_lookup(params=self.rel_embeddings, ids=self.neg_r)

        if config.L1_flag: # 如果L1_flag 就 使用 abs
            pos = tf.reduce_sum(input_tensor=abs(pos_h_e + pos_r_e - pos_t_e), axis=1, keepdims=True)
            neg = tf.reduce_sum(input_tensor=abs(neg_h_e + neg_r_e - neg_t_e), axis=1, keepdims=True)
            self.predict = pos
        else:
            pos = tf.reduce_sum(input_tensor=(pos_h_e + pos_r_e - pos_t_e) ** 2, axis=1, keepdims=True)
            neg = tf.reduce_sum(input_tensor=(neg_h_e + neg_r_e - neg_t_e) ** 2, axis=1, keepdims=True)
            self.predict = pos

        with tf.compat.v1.name_scope("output"):
            self.loss = tf.reduce_sum(input_tensor=tf.maximum(pos - neg + margin, 0))


def main(_):
    config = Config()
    if (config.testFlag):  # 暂时是false 不考虑
        test_lib.init()
        config.relation = test_lib.getRelationTotal() # 获取 关系数量
        config.entity = test_lib.getEntityTotal() # 获取实体数量
        config.batch = test_lib.getEntityTotal() # bitch size 就是 实体的总数
        config.batch_size = config.batch
    else:
        lib.init()
        config.relation = lib.getRelationTotal() # 获取 关系数量
        config.entity = lib.getEntityTotal()  # 获取实体数量
        config.batch_size = lib.getTripleTotal()  #bitch 就是三元组的总数

    with tf.Graph().as_default():
        sess = tf.compat.v1.Session()
        with sess.as_default():
            initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution=(
                "uniform" if False else "truncated_normal"))
            with tf.compat.v1.variable_scope("model", reuse=None, initializer=initializer): # 定义名称为模型
                trainModel = TransEModel(config=config)   # 定义模型主体

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(config.learning_rate) # 随机梯度下降
            grads_and_vars = optimizer.compute_gradients(trainModel.loss) # loss
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.initialize_all_variables())
            if (config.loadFromData):
                saver.restore(sess, 'model.vec')

            def train_step(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
                feed_dict = {
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                    trainModel.neg_h: neg_h_batch,
                    trainModel.neg_t: neg_t_batch,
                    trainModel.neg_r: neg_r_batch
                }
                _, step, loss = sess.run(
                    [train_op, global_step, trainModel.loss], feed_dict)
                return loss

            def test_step(pos_h_batch, pos_t_batch, pos_r_batch):
                feed_dict = {
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                }
                step, predict = sess.run(
                    [global_step, trainModel.predict], feed_dict)
                return predict

            ph = np.zeros(config.batch_size, dtype=np.int32)
            pt = np.zeros(config.batch_size, dtype=np.int32)
            pr = np.zeros(config.batch_size, dtype=np.int32)
            nh = np.zeros(config.batch_size, dtype=np.int32)
            nt = np.zeros(config.batch_size, dtype=np.int32)
            nr = np.zeros(config.batch_size, dtype=np.int32)

            # 获取地址
            ph_addr = ph.__array_interface__['data'][0]
            pt_addr = pt.__array_interface__['data'][0]
            pr_addr = pr.__array_interface__['data'][0]
            nh_addr = nh.__array_interface__['data'][0]
            nt_addr = nt.__array_interface__['data'][0]
            nr_addr = nr.__array_interface__['data'][0]

            lib.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            test_lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            test_lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            test_lib.testHead.argtypes = [ctypes.c_void_p]
            test_lib.testTail.argtypes = [ctypes.c_void_p]

            if not config.testFlag:
                for times in range(config.trainTimes):
                    res = 0.0
                    for batch in range(config.nbatches):
                        lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)
                        res += train_step(ph, pt, pr, nh, nt, nr)
                        current_step = tf.compat.v1.train.global_step(sess, global_step)
                    print(times)
                    print(res)
                saver.save(sess, 'model.vec')
            else:
                total = test_lib.getTestTotal()
                for times in range(total):
                    test_lib.getHeadBatch(ph_addr, pt_addr, pr_addr)
                    res = test_step(ph, pt, pr)
                    test_lib.testHead(res.__array_interface__['data'][0])

                    test_lib.getTailBatch(ph_addr, pt_addr, pr_addr)
                    res = test_step(ph, pt, pr)
                    test_lib.testTail(res.__array_interface__['data'][0])
                    print(times)
                    if (times % 50 == 0):
                        test_lib.test()
                test_lib.test()


if __name__ == "__main__":
    tf.compat.v1.app.run()
