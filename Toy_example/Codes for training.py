# 78epoch 达到52.16% 0.1123  state of the art
#18 9L_384 
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os

M=np.load('Index_Matrix.npy')
X=np.load('sample_X.npy')
Y=np.load('sample_Y.npy')
#1.labels  2.tanh 3.pick/drop/10
Adj_Matrix=np.load('Dynamic_Affinity_final_05.npy') #导入邻接矩阵
#修改了邻接矩阵
#Adj_Matrix=np.float32(Adj_Matrix[0,0])
Adj_Matrix.astype(np.float32)
#载入训练数据y标签
Dim_o3=18
y_data=np.load('Y_train_16p_mod.npy')  
y_data_3=np.load('Y_cluster_'+str(Dim_o3)+'.npy')#输入一共33个cluster的y值
external_num=9
X_prior=np.load('X_prior.npy')
X_pop=np.load('X_pop.npy')
y_data=np.where(y_data>0,y_data*2,y_data)

external_factor_1=np.load('External_1.npy')
external_factor_2=np.load('External_2.npy')
external_factor_3=np.load('External_3.npy')
Train_group1=np.load('Train_group1_mod.npy')
Train_group2=np.load('Train_group2_mod.npy')
Train_group3=np.load('Train_group3_mod.npy')
attribute_num=Train_group1.shape[2]

input1=tf.placeholder(shape=(None, 354, attribute_num),dtype=tf.float32)
input2=tf.placeholder(shape=(None, 354, attribute_num),dtype=tf.float32)
input3=tf.placeholder(shape=(None, 354, attribute_num),dtype=tf.float32)
input_external1=tf.placeholder(shape=(None, external_num),dtype=tf.float32)
input_external2=tf.placeholder(shape=(None, external_num),dtype=tf.float32)
input_external3=tf.placeholder(shape=(None, external_num),dtype=tf.float32)
X_prior_tensor=tf.placeholder(shape=(None, 354),dtype=tf.float32)
X_pop_tensor=tf.placeholder(shape=(None, 354),dtype=tf.float32)
Adj_Matrix_input1=tf.placeholder(shape=(None,354, 354),dtype=tf.float32)
Adj_Matrix_input2=tf.placeholder(shape=(None,354, 354),dtype=tf.float32)
Adj_Matrix_input3=tf.placeholder(shape=(None,354, 354),dtype=tf.float32)
#定义labels

y_true = tf.placeholder(shape=(None, 354,2), dtype=tf.float32)
#y_true2 = tf.placeholder(shape=(None, 354, 1), dtype=tf.float32)
y_true_o3= tf.placeholder(shape=(None, Dim_o3,1), dtype=tf.float32)  

####################
def weight_variable(shape):
    initial=tf.truncated_normal(shape,mean=0,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.truncated_normal(shape,mean=0,stddev=0.1)
    return tf.Variable(initial)

N_num=456
#2层网络加权
def gen_gcn_weights():
    """
    :return:
    """
    gcn_weights = {}
    N_num=456
    gcn_weights['gcn1'] = tf.Variable(tf.random_normal(shape=(6, N_num), mean=0, stddev=0.1))
    gcn_weights['gcn2'] = tf.Variable(tf.random_normal(shape=(N_num, N_num), mean=0, stddev=0.1))
    gcn_weights['gcn3'] = tf.Variable(tf.random_normal(shape=(N_num, N_num), mean=0, stddev=0.1))
    gcn_weights['gcn4'] = tf.Variable(tf.random_normal(shape=(N_num, N_num), mean=0, stddev=0.1))
    gcn_weights['gcn5'] = tf.Variable(tf.random_normal(shape=(N_num, N_num), mean=0, stddev=0.1))
    gcn_weights['gcn6'] = tf.Variable(tf.random_normal(shape=(N_num, N_num), mean=0, stddev=0.1))
    gcn_weights['gcn7'] = tf.Variable(tf.random_normal(shape=(N_num, N_num), mean=0, stddev=0.1))
    gcn_weights['gcn8'] = tf.Variable(tf.random_normal(shape=(N_num, N_num), mean=0, stddev=0.1))
    gcn_weights['gcn9'] = tf.Variable(tf.random_normal(shape=(N_num, 1), mean=0, stddev=0.1))
    gcn_weights['fusion'] = tf.Variable(tf.random_normal(shape=(3, 1), mean=0, stddev=0.1))
    gcn_weights['fusion2'] = tf.Variable(tf.random_normal(shape=(2, 1), mean=0, stddev=0.1))
    gcn_weights['fusion3'] = tf.Variable(tf.random_normal(shape=(64, 32), mean=0, stddev=0.1))
    gcn_weights['fusion3'] = tf.Variable(tf.random_normal(shape=(64, 32), mean=0, stddev=0.1))
    gcn_weights['embeds'] = tf.Variable(tf.random_normal(shape=(18, 6), mean=0, stddev=0.1))
    gcn_weights['high_low'] = tf.Variable(tf.random_normal(shape=(256, 128), mean=0, stddev=0.1))
    return gcn_weights
weights=gen_gcn_weights()


def GCN_1(input1,input_external,ex_num,Adj_M):
    #embedding
    input1_e = tf.map_fn(lambda x: tf.nn.leaky_relu(tf.matmul(x,weights['embeds']),alpha=0.8,name=None), input1)
    #定义第一层网络结构 GCN
    #layer1_temp = tf.map_fn(lambda x: tf.matmul(Adj_M, x), input1) #对于input1中的每个部分 都和A_tensor左乘
    layer1_temp = tf.matmul(Adj_M, input1_e)
    layer_1_output = tf.map_fn(lambda x: tf.matmul(x, weights['gcn1']), layer1_temp)
    print(layer_1_output.shape)
 #定义第二层网络结构 GCN
    #layer2_temp = tf.map_fn(lambda x: tf.matmul(Adj_M, x), layer_1_output)
    layer2_temp = tf.matmul(Adj_M, layer_1_output)
    layer_2_output = tf.map_fn(lambda x: tf.matmul(x, weights['gcn2']), layer2_temp)
  #  layer_2_output=tf.nn.elu(layer_2_output)
   #layer_con2_5 = tf.map_fn(lambda x: tf.matmul(x, weights['fusion3']), layer_2_output)
    layer_2_output=tf.nn.leaky_relu(layer_2_output,alpha=0.8, name=None)    

    img_shape = [4,354,64]
    #Wx_plus_b = tf.Variable(tf.random_normal(img_shape))
    axis = list(range(len(img_shape) - 1))
    wb_mean, wb_var = tf.nn.moments(layer_2_output, [0,1])
    scale = tf.Variable(tf.ones([N_num]))
    offset = tf.Variable(tf.zeros([N_num]))
    variance_epsilon = 0.001
    
    layer_2_output = tf.nn.batch_normalization(layer_2_output, wb_mean, wb_var, offset, scale, variance_epsilon)
    
    #定义第三层网络结构 GCN
    #layer3_temp = tf.map_fn(lambda x: tf.matmul(Adj_M, x), layer_2_output)
    layer3_temp = tf.matmul(Adj_M, layer_2_output)
    layer_3_output = tf.map_fn(lambda x: tf.matmul(x, weights['gcn3']), layer3_temp)
   # layer_3_output=tf.nn.leaky_relu(layer_3_output,alpha=0.8, name=None)
    #定义第四层网络结构 GCN
    #layer4_temp = tf.map_fn(lambda x: tf.matmul(Adj_M, x), layer_3_output)
    layer4_temp = tf.matmul(Adj_M, layer_3_output)
    layer_4_output = tf.map_fn(lambda x: tf.matmul(x, weights['gcn4']), layer4_temp)
    layer_4_output = tf.nn.leaky_relu(layer_4_output,alpha=0.8, name=None)
    
    axis = list(range(len(img_shape) - 1))
    wb_mean, wb_var = tf.nn.moments(layer_4_output, [0,1])
    scale = tf.Variable(tf.ones([N_num]))
    offset = tf.Variable(tf.zeros([N_num]))
    variance_epsilon = 0.001
    layer_4_output = tf.nn.batch_normalization(layer_4_output, wb_mean, wb_var, offset, scale, variance_epsilon)

    
 #   layer_2_output1=tf.map_fn(lambda x: tf.matmul(x, weights['high_low']), layer_2_output)
 #   output=tf.nn.leaky_relu(layer_4_output,alpha=0.8, name=None)
    #定义第五层网络结构 GCN
    layer5_temp = tf.matmul(Adj_M, layer_4_output)
    layer_5_output = tf.map_fn(lambda x: tf.matmul(x, weights['gcn5']), layer5_temp)

    

    layer6_temp = tf.matmul(Adj_M, layer_5_output)
    layer_6_output = tf.map_fn(lambda x: tf.matmul(x, weights['gcn6']), layer6_temp)    
    layer_6_output=tf.nn.leaky_relu(layer_6_output,alpha=0.8, name=None)

    scale = tf.Variable(tf.ones([N_num]))
    offset = tf.Variable(tf.zeros([N_num]))
    wb_mean, wb_var = tf.nn.moments(layer_6_output, [0,1])
    variance_epsilon = 0.001
    layer_6_output = tf.nn.batch_normalization(layer_6_output, wb_mean, wb_var, offset, scale, variance_epsilon)
  
    
    #定义第7层网络结构 GCN
    layer7_temp = tf.matmul(Adj_M, layer_6_output)
    layer_7_output = tf.map_fn(lambda x: tf.matmul(x, weights['gcn7']), layer7_temp)

    layer8_temp = tf.matmul(Adj_M, layer_7_output)
    layer_8_output = tf.map_fn(lambda x: tf.matmul(x, weights['gcn8']), layer8_temp)    
    layer_8_output=tf.nn.leaky_relu(layer_8_output,alpha=0.8, name=None)
    scale = tf.Variable(tf.ones([N_num]))
    offset = tf.Variable(tf.zeros([N_num]))  
    wb_mean, wb_var = tf.nn.moments(layer_8_output, [0,1])
    layer_8_output = tf.nn.batch_normalization(layer_8_output, wb_mean, wb_var, offset, scale, variance_epsilon)
    layer9_temp = tf.matmul(Adj_M, layer_8_output)
    layer_9_output = tf.map_fn(lambda x: tf.matmul(x, weights['gcn9']), layer9_temp)    
    layer_9_output=tf.nn.leaky_relu(layer_9_output,alpha=0.8, name=None)
    #output= tf.squeeze(layer_8_output , -1) 

    output= tf.squeeze(layer_9_output , -1) 

    
 #   output=tf.nn.tanh(output)
    print(output.shape)
  #  print(weights['gcn2'].shape)
    #bn3=tf.layers.batch_normalization(output, training=is_train)
    keep_prob=0.5
    #为全连接网络定义weights bias
    #keep_prob=tf.placeholder(tf.float32)
  #  input_external01=tf.expand_dims(input_external,1)
    def weight_variable(shape):
        initial=tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial=tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)
    
    
    External_W=weight_variable([ex_num,354])  
    External_b=bias_variable([354])
    # 接收到的输出值变成一维数据
    #External_=tf.reshape(X_pop_tensor,[-1,354]) #[n_samples,7,7,64]->[n_samples,7*7*64] #h表示hiddenlayer
    External_out=tf.nn.leaky_relu(tf.matmul(input_external1,External_W)+External_b,alpha=0.8, name=None)
 #   print("EXTE:",External_out.shape)
    
#    External_out=tf.expand_dims(External_out,-1)
    output=output+External_out
    output = tf.expand_dims(output,-1)
    print(output.shape)
    return output

ex_num=9
h_fc_out1=GCN_1(input1,input_external1,ex_num,Adj_Matrix_input1)
h_fc_out2=GCN_1(input2,input_external2,ex_num,Adj_Matrix_input2)
h_fc_out3=GCN_1(input3,input_external3,ex_num,Adj_Matrix_input3)
keep_prob=0.5

#分成三组
out1_out2=tf.concat([h_fc_out1, h_fc_out2], -1)  
out1_out3=tf.concat([out1_out2, h_fc_out3], -1)
# #w2 = tf.Variable(tf.truncated_normal(shape=(1,3,1),dtype=tf.float32,name='w2'))#shape=(filter_width,in_channels,out_channels) #做1*1的卷积
out1_out3 = tf.map_fn(lambda x: tf.matmul(x, weights['fusion']), out1_out3)

#加天气
external_num=9
ex_num=9
#func layer 接收到的输出值变成一维数据 天气external 两层全连接输出
W_fc_ex=weight_variable([external_num,354])  
b_fc_ex=bias_variable([354])
# 接收到的输出值变成一维数据
h_pool_ex_flat=tf.reshape(input_external3,[-1,external_num]) #[n_samples,7,7,64]->[n_samples,7*7*64] #h表示hiddenlayer
#h_fc_ex=tf.nn.leaky_relu(tf.matmul(h_pool_ex_flat,W_fc_ex)+b_fc_ex,alpha=0.8,name=None)
h_fc_ex=tf.nn.leaky_relu(tf.matmul(h_pool_ex_flat,W_fc_ex)+b_fc_ex,alpha=0.8,name=None)
out1_out3= tf.squeeze(out1_out3, -1)   
out_ex=h_fc_ex+out1_out3

###BN
print("out_ex:",out_ex.shape)
# wb_mean, wb_var = tf.nn.moments(out_ex, [1])
# scale = tf.Variable(tf.ones([1]))
# offset = tf.Variable(tf.zeros([1]))
# variance_epsilon = 0.001
# out_ex = tf.nn.batch_normalization(out_ex, wb_mean, wb_var, offset, scale, variance_epsilon)

#func layer 接收到的输出值变成一维数据 人口
W_pop=weight_variable([354,354])  
b_pop=bias_variable([354])
# 接收到的输出值变成一维数据
h_pool_ex_flat1=tf.reshape(X_pop_tensor,[-1,354]) #[n_samples,7,7,64]->[n_samples,7*7*64] #h表示hiddenlayer
h_fc_pop=tf.nn.leaky_relu(tf.matmul(h_pool_ex_flat1,W_pop)+b_pop,alpha=0.8, name=None)



##引入1层先验
W_fc_prior01=weight_variable([354,354])  
b_fc_prior01=bias_variable([354])
h_prior01=tf.reshape(X_prior_tensor,[-1,354])
h_prior01=tf.nn.leaky_relu(tf.matmul(h_prior01,W_fc_prior01)+b_fc_prior01,alpha=0.8, name=None)




#out1_out4 = tf.map_fn(lambda x: tf.matmul(x, weights['fusion2']), out3)
out1_out04=out_ex+2*h_fc_pop
#out_ex=tf.expand_dims(out_ex, -1)
#h_fc_pop=tf.expand_dims(h_fc_pop,-1)
#out1_out04=tf.concat([out_ex, h_fc_pop], -1)
#out1_out04 = tf.map_fn(lambda x: tf.matmul(x, weights['fusion2']), out1_out04)

W_task_2=weight_variable([354,354])  
b_task_2=bias_variable([354])


task_02_o=tf.nn.elu(tf.matmul(out1_out04,W_task_2)+b_task_2)
W_task_22=weight_variable([354,354])  
#b_task_22=weight_variable([354]) 
task_02_o2=tf.nn.elu(tf.matmul(task_02_o,W_task_22))
task_02_o2 = tf.expand_dims(task_02_o2, -1)

out1_out4 = out1_out04 + 0.5*h_prior01

scale = tf.Variable(tf.ones([1]))
offset = tf.Variable(tf.zeros([1]))
wb_mean, wb_var = tf.nn.moments(out1_out4, [0,1])
variance_epsilon = 0.001
out1_out4 = tf.nn.batch_normalization(out1_out4, wb_mean, wb_var, offset, scale, variance_epsilon)
    
    
    
#out1_out4  = out1_out04
#multi-task
W_task_1=weight_variable([354,354])  
b_task_1=bias_variable([354])

task_01_o=tf.nn.leaky_relu(tf.matmul(out1_out4,W_task_1)+b_task_1,alpha=0.8, name=None)
W_task_11=weight_variable([354,354]) 
#b_task_11=bias_variable([354])
task_01_o0 = tf.nn.leaky_relu(tf.matmul(task_01_o,W_task_11),alpha=0.8, name=None)



#img_shape = [4,354,1]
#Wx_plus_b = tf.Variable(tf.random_normal(img_shape))
#axis = list(range(len(img_shape) - 1))
#wb_mean, wb_var = tf.nn.moments(task_01_o0, [0,1])
#scale = tf.Variable(tf.ones([1]))
#offset = tf.Variable(tf.zeros([1]))
#variance_epsilon = 0.001
#task_01_o0 = tf.nn.batch_normalization(task_01_o0, wb_mean, wb_var, offset, scale, variance_epsilon)


W_task_3=weight_variable([354,Dim_o3])  
b_task_3=bias_variable([Dim_o3])

task_03_o30=tf.nn.leaky_relu(tf.matmul(task_01_o0,W_task_3)+b_task_3,alpha=1,name=None)

W_task_31=weight_variable([Dim_o3,Dim_o3])  
b_task_31=bias_variable([Dim_o3])
task_03_o30=tf.nn.relu(tf.matmul(task_03_o30,W_task_31)+b_task_31)
# task_03_o=tf.matmul(task_03_o,W_task_31)+b_task_31

# W_task_32=weight_variable([30,Dim_o3]) 
# #b_task_11=bias_variable([354])
# task_03_o3 = tf.nn.relu(tf.matmul(task_03_o,W_task_32))
task_03_o3 = tf.expand_dims(task_03_o30, -1)

#task_03_o3 = tf.cast(task_03_o3,dtype=tf.int16)

###task_o3的输出再反过来输入到task1中
W_task_3_1=weight_variable([Dim_o3,354])  
b_task_3_1=bias_variable([354])

task_03_31=tf.matmul(task_03_o30,W_task_3_1)+b_task_3_1

task_01_o0=task_01_o0+task_03_31
task_01_o1 = tf.expand_dims(task_01_o0, -1)

#####输入完毕，加 完成


y_true1=y_true[:,:,0]
y_true1=tf.expand_dims(y_true1,-1)
y_true2=y_true[:,:,1]
y_true2=tf.expand_dims(y_true2,-1)
y_true3=y_true_o3[:,:,0]
y_true3=tf.expand_dims(y_true3,-1)

#current_epoch = tf.Variable(0)
reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4),tf.trainable_variables())

loss =1.2*tf.reduce_mean(tf.square(task_01_o1 - y_true1))+0.8*tf.reduce_mean(tf.square(task_02_o2 - y_true2))+reg+tf.reduce_mean(tf.square(task_03_o3 - y_true3))
#+cross_entropy

#定义学习率
learning_rate = 0.0005
#训练方式

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#导入数据
X_features1=Train_group1
X_prior_external1=external_factor_1
X_features2=Train_group2
X_prior_external2=external_factor_2
X_features3=Train_group3
X_prior_external3=external_factor_3




labels=y_data[:,:,0:2]
#Data split
Total=7000
tra=4800
test=100

#处理梯度
epoch = 200
batch_size = 300
n_batch = int(tra/ batch_size)
# 把batch分成多少个sub batch来计算
subdivisions = 20
subdivisions_batch_size = int(np.ceil(batch_size / subdivisions))
print(epoch, batch_size, n_batch, subdivisions, subdivisions_batch_size)
optim = tf.train.AdamOptimizer(learning_rate)
grads_vars = optim.compute_gradients(loss)
# 删掉没梯度的参数, 倒序删除，减少麻烦
for i in range(len(grads_vars))[::-1]:
    if grads_vars[i][0] is None:
        del grads_vars[i]
# 生成梯度缓存
grads_cache = [tf.Variable(np.zeros(t[0].shape.as_list(), np.float32), trainable=False) for t in grads_vars]
# 清空梯度缓存op，每一 batch 开始前调用
clear_grads_cache_op = tf.group([gc.assign(tf.zeros_like(gc)) for gc in grads_cache])
# 累积梯度op，累积每个 sub batch 的梯度
accumulate_grad_op = tf.group([gc.assign_add(gv[0]) for gc, gv in zip(grads_cache, grads_vars)])
# 求平均梯度，
mean_grad = [gc / tf.cast(subdivisions,dtype=tf.float32) for gc in grads_cache]
# 组装梯度列表
new_grads_vars = [(g, gv[1]) for g, gv in zip(mean_grad, grads_vars)]
# 应用梯度op，累积完所有 sub batch 的梯度后，应用梯度
apply_grad_op = optim.apply_gradients(new_grads_vars)

#开始训练
import datetime
#import evaluation
saver = tf.train.Saver()
epoch_loss = []
starttime = datetime.datetime.now()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        print(i)
        loss_sum = 0
        current_epoch=i
        for b in range(n_batch):
            print("b:",b)
            y_batch = labels[b * batch_size:(b + 1) * batch_size]
            y_batch_3 = y_data_3[b * batch_size:(b + 1) * batch_size]
            X_batch_in1 = X_features1[b * batch_size:(b + 1) * batch_size]
            X_batch_in2 = X_features2[b * batch_size:(b + 1) * batch_size]
            X_batch_in3 = X_features3[b * batch_size:(b + 1) * batch_size]
            X_external_batch1 = X_prior_external1[b * batch_size:(b + 1) * batch_size]
            X_external_batch2 = X_prior_external2[b * batch_size:(b + 1) * batch_size]
            X_external_batch3 = X_prior_external3[b * batch_size:(b + 1) * batch_size]
            X_prior_batch=X_prior[b * batch_size:(b + 1) * batch_size]
            X_pop_batch=X_pop[b * batch_size:(b + 1) * batch_size]
            sess.run(clear_grads_cache_op)
            sub_loss_sum = 0
            for s in range(subdivisions):
                #print("s:",s)
                y_sub_batch = y_batch[s * subdivisions_batch_size:(s + 1) * subdivisions_batch_size]
                y_sub_batch_3 = y_batch_3[s * subdivisions_batch_size:(s + 1) * subdivisions_batch_size]
                X_sub_batch_in1 = X_batch_in1[s * subdivisions_batch_size:(s + 1) * subdivisions_batch_size]
                X_sub_batch_in2 = X_batch_in2[s * subdivisions_batch_size:(s + 1) * subdivisions_batch_size]
                X_sub_batch_in3 = X_batch_in3[s * subdivisions_batch_size:(s + 1) * subdivisions_batch_size]
                X_external_sub_batch1 = X_external_batch1[s * subdivisions_batch_size:(s + 1) * subdivisions_batch_size]
                X_external_sub_batch2 = X_external_batch2[s * subdivisions_batch_size:(s + 1) * subdivisions_batch_size]
                X_external_sub_batch3 = X_external_batch3[s * subdivisions_batch_size:(s + 1) * subdivisions_batch_size]
                X_prior_sub_batch = X_prior_batch[s * subdivisions_batch_size:(s + 1) * subdivisions_batch_size]
                X_pop_sub_batch = X_pop_batch[s * subdivisions_batch_size:(s + 1) * subdivisions_batch_size]
                Matrix_1=np.zeros([subdivisions_batch_size,354,354])
                Matrix_2=np.zeros([subdivisions_batch_size,354,354])
                Matrix_3=np.zeros([subdivisions_batch_size,354,354])
                p=0
                for k in range(s * subdivisions_batch_size,(s + 1) * subdivisions_batch_size):    
                        Matrix_1[p,:,:]=Adj_Matrix[M[X[k,1],0],M[X[k,1],1],:,:]
                        Matrix_2[p,:,:]=Adj_Matrix[M[X[k,4],0],M[X[k,4],1],:,:]
                        Matrix_3[p,:,:]=Adj_Matrix[M[X[k,7],0],M[X[k,7],1],:,:]
                        p=p+1

                if len(y_sub_batch) == 0:
                    break
                feed_dict = {input1:X_sub_batch_in1,input2:X_sub_batch_in2,input3:X_sub_batch_in3,
                             X_prior_tensor:X_prior_sub_batch,X_pop_tensor:X_pop_sub_batch,input_external1:X_external_sub_batch1,
                             input_external2:X_external_sub_batch2,input_external3:X_external_sub_batch3,y_true: y_sub_batch,y_true_o3:y_sub_batch_3,Adj_Matrix_input1:Matrix_1,Adj_Matrix_input2:Matrix_2,Adj_Matrix_input3:Matrix_3}
                _, los = sess.run([accumulate_grad_op, loss], feed_dict)
                
                sub_loss_sum += los
            loss_sum += sub_loss_sum / subdivisions
            # print('sub_loss', sub_loss_sum / subdivisions)
            # 梯度累积完成，开始应用梯度
            sess.run(apply_grad_op)
        print('loss', loss_sum / n_batch)
        epoch_loss.append(loss_sum / n_batch)
        endtime = datetime.datetime.now()
        print((endtime - starttime).seconds)
        # 预测
        if i%1==0:
            los0=0
            pred_Y1 = []
            pred_Y2 = []
            pred_Y4 = []
            pred_Y3 = []
            for j in range(tra,tra+test):
                X_feature1_test = X_features1[j:j+1]
                X_feature2_test = X_features2[j:j+1]
                X_feature3_test = X_features3[j:j+1]
                X_external1_test= X_prior_external1[j:j+1]
                X_external2_test= X_prior_external2[j:j+1]
                X_external3_test= X_prior_external3[j:j+1]   
                X_prior_test=X_prior[j:j+1]
                X_pop_test=X_pop[j:j+1]
                y_label_test=labels[j:j+1]
                y_label_test_3=y_data_3[j:j+1]
                Adj_test1=Adj_Matrix[M[X[j,1],0],M[X[j,1],1],:,:].reshape([1,354,354])
                Adj_test2=Adj_Matrix[M[X[j,4],0],M[X[j,4],1],:,:].reshape([1,354,354])
                Adj_test3=Adj_Matrix[M[X[j,7],0],M[X[j,7],1],:,:].reshape([1,354,354])

                feed_dict = {input1: X_feature1_test,input2:X_feature2_test,input3:X_feature3_test,X_prior_tensor:X_prior_test,X_pop_tensor:X_pop_test,input_external1:X_external1_test,input_external2:X_external2_test,input_external3:X_external3_test,y_true:y_label_test,y_true_o3:y_label_test_3,
                            Adj_Matrix_input1:Adj_test1,Adj_Matrix_input2:Adj_test2,Adj_Matrix_input3:Adj_test3                         
                            }
                pred1 = sess.run(task_01_o1, feed_dict=feed_dict) #X_prior_tensor:X_prior input_external3:X_external3
                pred2 = sess.run(task_02_o2, feed_dict=feed_dict) 
                pred3 = sess.run(task_03_o3, feed_dict=feed_dict) 
              
                pred_Y1.append(pred1)
                pred_Y2.append(pred2)
                pred_Y3.append(pred3)
                los1= sess.run([train_op, loss], feed_dict=feed_dict)
                los0=los0+np.array(los1)[1]
                #np.save(''pred_Y1)
            #print('val_los:',sess.run(tf.reduce_mean(tf.square(pred_Y- y_label_test))))
                #print('task_1_val_los:',sess.run(tf.reduce_mean(tf.square(pred_Y1- y_true1))))
                #print('task_2_val_los:',sess.run(tf.reduce_mean(tf.square(pred_Y2- y_true2))))
            #np.save('GCN_results_1/Result_STMGAP_4_'+str(i)+'.npy',pred_Y) 
            #Acc1=evaluation.calculate_acc(np.array(pred_Y),np.array(pred_Y4))
            #Acc2=evaluation.calculate_acc(np.array(pred_Y2),np.array(pred_Y4))
            #print(Acc1,Acc2)
            np.save('GCN_results_07/task_1_0814_4800_100_'+str(Dim_o3)+'_9L_456_'+str(i)+'.npy',pred_Y1) 
            np.save('GCN_results_07/task_3_0814_4800_100_'+str(Dim_o3)+'_9L_456_'+str(i)+'.npy',pred_Y3) 


