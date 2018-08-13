import tensorflow as tf
import numpy as np
from PSNR import psnr
import os,re,glob
import scipy.io
import pickle
import time
from tensorflow.core.protobuf import saver_pb2

class VDSR():
    def __init__(self,sess, flags):
        self.sess=sess
        self.flags = flags

    def model(self, input_tensor):
        with tf.device("/gpu:0"):
            weights = []
            tensor = None

            conv_w_input = tf.get_variable("convW_input", [3,3,1,64], initializer=tf.contrib.layers.xavier_initializer())
            conv_b_input = tf.get_variable("convb_input", [64], initializer=tf.contrib.layers.xavier_initializer())
            weights.append(conv_w_input)
            weights.append(conv_b_input)
            tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_w_input, strides=[1,1,1,1], padding='SAME'), conv_b_input))

            for i in range(self.flags.layer):
                conv_w = tf.get_variable("conv_%02d_w" % (i+1), [3,3,64,64],initializer=tf.contrib.layers.xavier_initializer())
                conv_b = tf.get_variable("conv_%02d_b" % (i+1), [64], initializer=tf.constant_initializer(0))
                weights.append(conv_w)
                weights.append(conv_b)
                tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))

            conv_w_output = tf.get_variable("convw_output",[3,3,64,1], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
            conv_b_output = tf.get_variable("convb_output", [1], initializer=tf.constant_initializer(0))
            weights.append(conv_w_output)
            weights.append(conv_b_output)
            tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w_output, strides=[1,1,1,1], padding='SAME'), conv_b_output)
            tensor = tf.add(tensor, input_tensor)
            return tensor, weights

    def get_train_list(self):
        l = glob.glob(os.path.join(self.flags.train_data_path,"*"))
        print(len(l))
        l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
        print(len(l))
        train_list = []
        for f in l:
            if os.path.exists(f):
                if os.path.exists(f[:-4]+"_2.mat"): train_list.append([f, f[:-4]+"_2.mat"])
                if os.path.exists(f[:-4]+"_3.mat"): train_list.append([f, f[:-4]+"_3.mat"])
                if os.path.exists(f[:-4]+"_4.mat"): train_list.append([f, f[:-4]+"_4.mat"])
        return train_list

    def get_image_batch(self,train_list,offset,batch_size):
        target_list = train_list[offset:offset+batch_size]
        input_list = []
        gt_list = []

        for pair in target_list:
            input_img = scipy.io.loadmat(pair[1])['patch']
            gt_img = scipy.io.loadmat(pair[0])['patch']
            input_list.append(input_img)
            gt_list.append(gt_img)
        input_list = np.array(input_list)
        input_list.resize([self.flags.batch_size, self.flags.image_size, self.flags.image_size, 1])
        gt_list = np.array(gt_list)
        gt_list.resize([self.flags.batch_size, self.flags.image_size, self.flags.image_size, 1])
        return input_list, gt_list

    def get_img_list(self,data_path):
        l = glob.glob(os.path.join(data_path,"*"))
        l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
        test_list=[]
        for f in l:
                    if os.path.exists(f):
                        if os.path.exists(f[:-4]+"_2.mat"): test_list.append([f, f[:-4]+"_2.mat", 2])
                        if os.path.exists(f[:-4]+"_3.mat"): test_list.append([f, f[:-4]+"_3.mat", 3])
                        if os.path.exists(f[:-4]+"_4.mat"): test_list.append([f, f[:-4]+"_4.mat", 4])
        return test_list


    def test_VDSR(self, epoch, ckpt_path, data_path):
        with tf.Session() as sess:
            self.test_VDSR_with_sess(epoch, ckpt_path, data_path, sess)

    def test_VDSR_with_sess(self, ckpt_path, data_path, output_tensor, saver):
        folder_list = glob.glob(os.path.join(data_path, 'Set*'))
        print('folder_list', folder_list)
        saver.restore(self.sess, ckpt_path)
        psnr_dict = {}
        for folder_path in folder_list:
            psnr_list = []
            img_list = self.get_img_list(folder_path)
            print('img_list',img_list)
            for i in range(len(img_list)):
                input_list, gt_list, scale_list = self.get_test_image(img_list, i,1)

                print(len(input_list))
                input_y = input_list[0]
                gt_y = gt_list[0]

                start_t = time.time()
                img_vdsr_y = self.sess.run([output_tensor], feed_dict={ self.input_tensor: np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], 1))})
                img_vdsr_y = np.resize(img_vdsr_y, (input_y.shape[0], input_y.shape[1]))
                end_t = time.time()
                print("end_t", end_t,"start_t", start_t)
                print("time consumption", end_t-start_t)
                print("image_Size", input_y.shape)
                psnr_bicub = psnr(input_y, gt_y, scale_list[0])
                psnr_vdsr = psnr(img_vdsr_y, gt_y, scale_list[0])
                print("bicubic PSNR: %f \t VDSR PSNR: %f" % (psnr_bicub, psnr_vdsr))
                psnr_list.append([psnr_bicub, psnr_vdsr, scale_list[0]])

            psnr_dict[os.path.basename(folder_path)] = psnr_list
        print("psnr_dict:", psnr_dict)
        with open('./psnr\%s' %os.path.basename(ckpt_path), 'wb') as f:
            pickle.dump(psnr_dict, f)

    def get_test_image(self, test_list, offset, batch_size):
        target_list = test_list[offset:offset+batch_size]
        input_list = []
        gt_list = []
        scale_list = []
        print('target_list',target_list)
        for pair in target_list:
            print('pair1',pair[0])
            print('pair2',pair[1])
            mat_dict = scipy.io.loadmat(pair[1])
            input_img = None
            print('mat_dict',mat_dict)

            if "img_2" in mat_dict: input_img = mat_dict["img_2"]
            elif "img_3" in mat_dict: input_img = mat_dict["img_3"]
            elif "img_4" in mat_dict: input_img = mat_dict["img_4"]
            else: continue

            gt_img = scipy.io.loadmat(pair[0])['img_raw']
            input_list.append(input_img)
            gt_list.append(gt_img)
            scale_list.append(pair[2])
        return input_list, gt_list, scale_list

    def train(self):
        if  self.flags.is_train:
            print("Start Training")
            train_list = self.get_train_list()
            self.train_input        = tf.placeholder(tf.float32, shape=(self.flags.batch_size, self.flags.image_size, self.flags.image_size, 1))
            self.train_groundtruth  = tf.placeholder(tf.float32, shape=(self.flags.batch_size, self.flags.image_size, self.flags.image_size, 1))
            shared_model = tf.make_template('shared_model', self.model)
            train_output, weights 	= shared_model(self.train_input)
            loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(train_output, self.train_groundtruth)))
            for w in weights:
                 loss += tf.nn.l2_loss(w)*1e-4
            tf.summary.scalar("loss", loss)

            global_step 	= tf.Variable(0, trainable=False)
            learning_rate 	= tf.train.exponential_decay(self.flags.learning_rate, global_step*self.flags.batch_size, len(train_list)*self.flags.learning_rate_step_size, self.flags.learning_rate_decay_rate, staircase=True)
            optimizer=tf.train.MomentumOptimizer(learning_rate, 0.9)
            tf.summary.scalar("learning rate", learning_rate)
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, tf.div(-0.1*self.flags.learning_rate, learning_rate), tf.div(0.1*self.flags.learning_rate,learning_rate)), var) for grad, var in gvs]
            opt = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

            if not os.path.exists('logs'):
                os.mkdir('logs')
            merged = tf.summary.merge_all()
            file_writer = tf.summary.FileWriter('logs', self.sess.graph)

            tf.initialize_all_variables().run()

            if self.flags.model_path:
                print ("restore model...")
                saver.restore(self.sess, self.flags.model_path)
                print("Done")

            for epoch in range(0, self.flags.max_epoch):
                for step in range(len(train_list)//self.flags.batch_size):
                    offset = step*self.flags.batch_size
                    input_data ,groudtruth_data = self.get_image_batch(train_list, offset, self.flags.batch_size)
                    feed_dict = {self.train_input: input_data, self.train_groundtruth: groudtruth_data}
                    _, l, output, current_learning_rate, g_step = self.sess.run([opt, loss, train_output, learning_rate, global_step], feed_dict=feed_dict)
                    print("[epoch %2.4f] loss %.4f\t lr %.5f"%(epoch+(float(step)*self.flags.batch_size/len(train_list)), np.sum(l)/self.flags.batch_size, current_learning_rate))
                    del input_data, groudtruth_data

                saver.save(self.sess, "./checkpoints/VDSR_const_clip_0.01_epoch_%03d.ckpt" % epoch ,global_step=global_step)

    def test(self):
            model_list= sorted(glob.glob("./checkpoints/VDSR_const_clip_0.01_epoch*"))
            model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("meta")]
            
            print(model_list)
            self.input_tensor = tf.placeholder(tf.float32, shape=(1,None,None,1))
            shared_model = tf.make_template('shared_model', self.model)
            output_tensor, weights = shared_model(self.input_tensor)
            saver= tf.train.Saver(weights)
            tf.initialize_all_variables().run()
            for model_ckpt in model_list:
                print(model_ckpt)
                epoch = int(model_ckpt.split('epoch_')[-1].split('.ckpt')[0])
                print("Testing_model", model_ckpt)
                self.test_VDSR_with_sess(model_ckpt, self.flags.test_img, output_tensor, saver)









