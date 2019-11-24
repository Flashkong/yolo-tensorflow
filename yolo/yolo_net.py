import numpy as np
import tensorflow as tf
import yolo.config as cfg

slim = tf.contrib.slim


class YOLONet(object):

    def __init__(self, is_training=True):
        ''' 构造函数
        利用 cfg 文件对网络参数进行初始化，同时定义网络的输入和输出 size 等信息，
        其中 offset 的作用应该是一个定长的偏移
        boundery1和boundery2 作用是在输出中确定每种信息的长度（如类别，置信度等）。
        其中 boundery1 指的是对于所有的 cell 的类别的预测的张量维度，所以是 self.cell_size * self.cell_size * self.num_class
        boundery2 指的是在类别之后每个cell 所对应的 bounding boxes 的数量的总和，所以是self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        args:
            is_training：训练？
        '''
        # VOC 2012数据集类别名
        self.classes = cfg.CLASSES
        # 类别个数C 20
        self.num_class = len(self.classes)
        # 网络输入图像大小448， 448 x 448
        self.image_size = cfg.IMAGE_SIZE
        # 单元格大小S=7  将图像分为SxS的格子
        self.cell_size = cfg.CELL_SIZE
        # 每个网格边界框的个数B=2
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        # 网络输出的大小 S*S*(B*5 + C) = 1470
        self.output_size = (self.cell_size * self.cell_size) * \
                           (self.num_class + self.boxes_per_cell * 5)
        # 图片的缩放比例 64
        self.scale = 1.0 * self.image_size / self.cell_size
        # 将网络输出分离为类别和置信度以及边界框的大小，输出维度为7*7*20 + 7*7*2 + 7*7*2*4=1470
        # 7*7*20
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        # 7*7*20+7*7*2
        self.boundary2 = self.boundary1 + \
                         self.cell_size * self.cell_size * self.boxes_per_cell

        # 代价函数 权重
        self.object_scale = cfg.OBJECT_SCALE  # 1
        self.noobject_scale = cfg.NOOBJECT_SCALE  # 1
        self.class_scale = cfg.CLASS_SCALE  # 2.0
        self.coord_scale = cfg.COORD_SCALE  # 5.0
        # 学习率0.0001
        self.learning_rate = cfg.LEARNING_RATE
        # batch大小 45
        self.batch_size = cfg.BATCH_SIZE
        # 泄露修正线性激活函数 系数0.1
        self.alpha = cfg.ALPHA
        # 偏置 形状[7,7,2]
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
        # 输入图片占位符 [NONE,image_size,image_size,3]
        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')
        # build_network把所有的结果输出了, 包括分类和坐标
        # logits = [None, 1470], 1470 = 7x7x30
        # 构建网络 获取YOLO网络的输出(不经过激活函数的输出)  形状[None,1470]
        self.logits = self.build_network(
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)

        if is_training:
            # 对于每一个图片，在训练的时候，经过网络输出的output应该是7*7*30，而每一张图片的标签应该是7*7*(5+20)
            # 因为每个图片只有GT的bbox和class分类结果
            # 如果是测试的话就不用做这个步骤，如果是训练的话就做计算loss的步骤
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class])
            # 设置损失函数
            self.loss_layer(self.logits, self.labels)
            # 加入权重正则化之后的损失函数
            self.total_loss = tf.losses.get_total_loss()
            # 将损失以标量形式显示，该变量命名为total_loss
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        # 定义变量命名空间
        with tf.variable_scope(scope):
            # 定义共享参数  使用l2正则化
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    activation_fn=leaky_relu(alpha),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            ):
                # 在图像的宽高两个维度上都pad 3，pad_1 填充 454x454x3
                net = tf.pad(
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                    name='pad_1')
                # 卷积层conv_2 s=2    (n-f+1)/s向上取整    224x224x64
                net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                # 池化层pool_3 112x112x64
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                # 卷积层conv_4、3x3x192 s=1  n/s向上取整   112x112x192
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                # 池化层pool_5 56x56x192
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                # 卷积层conv_6、1x1x128 s=1  n/s向上取整  56x56x128
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                # 卷积层conv_7、3x3x256 s=1  n/s向上取整 56x56x256
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                # 卷积层conv_8、1x1x256 s=1  n/s向上取整 56x56x256
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                # 卷积层conv_9、3x3x512 s=1  n/s向上取整 56x56x512
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                # 池化层pool_10 28x28x512
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                # 卷积层conv_11、1x1x256 s=1  n/s向上取整 28x28x256
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                # 卷积层conv_12、3x3x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                # 卷积层conv_13、1x1x256 s=1  n/s向上取整 28x28x256
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                # 卷积层conv_14、3x3x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                # 卷积层conv_15、1x1x256 s=1  n/s向上取整 28x28x256
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                # 卷积层conv_16、3x3x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                # 卷积层conv_17、1x1x256 s=1  n/s向上取整 28x28x256
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                # 卷积层conv_18、3x3x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                # 卷积层conv_19、1x1x512 s=1  n/s向上取整 28x28x512
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                # 卷积层conv_20、3x3x1024 s=1  n/s向上取整 28x28x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                # 池化层pool_21 14x14x1024
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                # 卷积层conv_22、1x1x512 s=1  n/s向上取整 14x14x512
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                # 卷积层conv_23、3x3x1024 s=1  n/s向上取整 14x14x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                # 卷积层conv_24、1x1x512 s=1  n/s向上取整 14x14x512
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                # 卷积层conv_25、3x3x1024 s=1  n/s向上取整 14x14x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                # 卷积层conv_26、3x3x1024 s=1  n/s向上取整 14x14x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                # pad_27 填充 16x16x2014
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')
                # 卷积层conv_28、3x3x1024 s=2  (n-f+1)/s向上取整 7x7x1024
                net = slim.conv2d(
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')
                # 卷积层conv_29、3x3x1024 s=1  n/s向上取整 7x7x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                # 卷积层conv_30、3x3x1024 s=1  n/s向上取整 7x7x1024
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                # 假设原来的大小为N*7*7*1024 转置之后的大小为N*1024*7*7
                # trans_31 转置[None,1024,7,7]
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                # 在经过flatten之后，变为N*50176 ，其中50176 = 1024*7*7
                # flat_32 展开 50176
                net = slim.flatten(net, scope='flat_32')
                # 全连接层fc_33  512
                net = slim.fully_connected(net, 512, scope='fc_33')
                # 全连接层fc_34  4096
                net = slim.fully_connected(net, 4096, scope='fc_34')
                # 弃权层dropout_35 4096
                net = slim.dropout(
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                # 全连接层fc_36 1470
                net = slim.fully_connected(
                    net, num_outputs, activation_fn=None, scope='fc_36')
        return net

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """
        calculate ious
        这个函数的主要作用是计算两个 bounding box 之间的 IoU。输入是两个 5 维的bounding box,输出的两个 bounding Box 的IoU

        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
          注意这里的参数x_center, y_center, w, h都是归一到[0,1]之间的，分别表示预测边界框的中心相对整张图片的坐标，宽和高
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            # intersection 这个就是求相交矩形的长和宽，所以有rd-ru，相当于x1-x2和y1-y2，
            # 之所以外面还要加一个tf.maximum是因为删除那些不合理的框，比如两个框没交集，
            # 就会出现左上角坐标比右下角还大。
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]
            # 外面有个tf.maximum这个就是保证相交面积不为0, 因为后面要做分母
            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
        # 最后有一个tf.clip_by_value,这个是将如果你的交并比大于1,那么就让它等于1,如果小于0,那么就
        # 让他变为0,因为交并比在0-1之间。
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        '''
            计算预测和标签之间的损失函数

            args：
                todo : 这里不太懂
                predicts：Yolo网络的输出 形状[None,1470]
                          0：7*7*20：表示预测类别
                          7*7*20:7*7*20 + 7*7*2:表示预测置信度，即预测的边界框与实际边界框之间的IOU
                          7*7*20 + 7*7*2：1470：预测边界框    目标中心是相对于当前格子的，宽度和高度的开根号是相对当前整张图像的(归一化的)
                labels：标签值 形状[None,7,7,25]
                          0:1：置信度，表示这个地方是否有目标
                          1:5：目标边界框  目标中心，宽度和高度(没有归一化)
                          5:25：目标的类别
        '''
        with tf.variable_scope(scope):
            # 将网络输出分离为类别和置信度以及边界框的大小，
            # 输出维度为7*7*20 + 7*7*2 + 7*7*2*4=1470
            # 预测每个格子目标的类别 形状[45,7,7,20]
            predict_classes = tf.reshape(
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            # 预测每个格子中两个边界框的置信度 形状[45,7,7,2]
            predict_scales = tf.reshape(
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            # 预测每个格子中的两个边界框，(x,y)表示边界框相对于格子边界框的中心 w,h的开根号相对于整个图片  形状[45,7,7,2,4]
            predict_boxes = tf.reshape(
                predicts[:, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
            # 标签的置信度,表示这个地方是否有框 形状[45,7,7,1]
            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])
            # 标签的边界框 (x,y)表示边界框相对于整个图片的中心 形状[45,7,7,1，4]
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            # 标签的边界框 归一化后 张量沿着axis=3重复两边，扩充后[45,7,7,2,4]
            boxes = tf.tile(
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[..., 5:]  # class经过这样的处理之后得到的结果是45*7*7*20 而不是（45*7*7）*20
            '''
            predict_boxes_tran：offset变量用于把预测边界框predict_boxes中的坐标中心(x,y)由相对当前格子转换为相对当前整个图片

            offset，这个是构造的[7,7,2]矩阵，每一行都是[7,2]的矩阵，值为[[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]]
            这个变量是为了将每个cell的坐标对齐，后一个框比前一个框要多加1
            比如我们预测了cell_size的每个中心点坐标，那么我们这个中心点落在第几个cell_size
            就对应坐标要加几，这个用法比较巧妙，构造了这样一个数组，让他们对应位置相加
            '''
            # offset shape为[1,7,7,2]  如果忽略axis=0，则每一行都是  [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]]
            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            # shape为[45,7,7,2]
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            # shape为[45,7,7,2]  如果忽略axis=0 第i行为[[i,i],[i,i],[i,i],[i,i],[i,i],[i,i],[i,i]]
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))
            # shape为[45,7,7,2,4]  计算每个格子中的预测边界框坐标(x,y)相对于整个图片的位置  而不是相对当前格子
            # 假设当前格子为(3,3)，当前格子的预测边界框为(x0,y0)，则计算坐标(x,y) = ((x0,y0)+(3,3))/7
            # todo 这里不懂为什么一个加的是offset而另一个加的是offset_tran
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + offset) / self.cell_size,
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                 tf.square(predict_boxes[..., 2]),
                 tf.square(predict_boxes[..., 3])], axis=-1)
            # 计算每个格子预测边界框与真实边界框之间的IOU  [45,7,7,2]
            # todo 这里输入的两个w和h应该一个是0-1的范围的，一个是开平方的，有问题
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # 这个是求论文中的1ijobj参数，[45,7,7,2]     1ijobj：表示网格单元i的第j个编辑框预测器’负责‘该预测
            # 先计算每个框交并比最大的那个，因为我们知道，YOLO每个格子预测两个边界框，一个类别。在训练时，每个目标只需要
            # 一个预测器来负责，我们指定一个预测器"负责"，根据哪个预测器与真实值之间具有当前最高的IOU来预测目标。
            # 所以object_mask就表示每个格子中的哪个边界框负责该格子中目标预测？哪个边界框取值为1，哪个边界框就负责目标预测
            # 当格子中的确有目标时，取值为[1,1]，[1,0],[0,1]
            # 比如某一个格子的值为[1,0]，表示第一个边界框负责该格子目标的预测  [0,1]：表示第二个边界框负责该格子目标的预测
            # 当格子没有目标时，取值为[0,0]
            # 其中，3代表[45，7，7，2]中，2这一个维度，相当于是在那两个边界框的IOU当中取了最大值，取完之后大小为，[45,7,7,1]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            # 相当于a是(45,7,7,2) b是(45,7,7,1) 那么a>=b得到的结果是(45,7,7,2)而且结果都是(True,False)这种类型的
            # 在这里又使用了一个类型转换的，把他们转换一下，乘以response就是相当于判断了一下这个cell是否有框
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # noobject_mask就表示每个边界框不负责该目标的置信度，
            # 使用tf.onr_like，使得全部为1,再减去有目标的，也就是有目标的对应坐标为1,这样一减，就变为没有的了。[45,7,7,2]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask
            # boxes_tran 这个就是把之前的坐标换回来(相对整个图像->相对当前格子)，长和宽开方(原因在论文中有说明)，后面求loss就方便。 shape为(4, 45, 7, 7, 2)
            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss 分类损失，如果目标出现在网格中 response为1，否则response为0  原文代价函数公式第5项
            # 该项表名当格子中有目标时，预测的类别越接近实际类别，代价值越小  原文代价函数公式第5项
            # class_delta 的 大小为[45*7*7*20]
            # 在这里，这个response就代表了1iobj，即if object appears in cell i
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale

            # object_loss
            # object_loss 有目标物体存在的置信度预测损失   原文代价函数公式第3项
            # 该项表名当格子中有目标时，负责该目标预测的边界框的置信度越越接近预测的边界框与实际边界框之间的IOU时，代价值越小
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss
            # noobject_loss  没有目标物体存在的置信度的损失(此时iou_predict_truth为0)  原文代价函数公式第4项
            # 该项表名当格子中没有目标时，预测的两个边界框的置信度越接近0，代价值越小
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss
            # coord_loss 边界框坐标损失 shape 为 [batch_size, 7, 7, 2, 1]  原文代价函数公式1,2项
            # 该项表名当格子中有目标时，预测的边界框越接近实际边界框，代价值越小
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale
            # 将所有损失放在一起
            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)
            # 将每个损失添加到日志记录
            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)


def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')

    return op
