import tensorflow as tf
from nets.resnet_colorizer import ResNetColorizer
import cv2
import numpy as np
import vot
from src.parse_arguments import parse_arguments
from PIL import Image

class ColorizationTracker:
    def __init__(self, imagepath, region):
        self.track_count = 0
        #Parameters
        self.exemplar_sz = 128
        self.search_sz = 256
        #Parse the arguments
        self.hp, self.evaluation, self.run, self.env, self.design = parse_arguments(mode='color')

        #Get first frame image and ground-truth
        self.region = region
        self.pos_x = region.x + region.width / 2
        self.pos_y = region.y + region.height / 2
        self.target_w = region.width
        self.target_h = region.height


        #Calculate the size of final score (upscaled size of score matrix, where score matrix
        # is convolution of results of two branches of siamese network)
        self.final_score_sz = self.hp.response_up * (self.design.score_sz - 1)

        #Calculate the scale factors
        self.scale_factors = self.hp.scale_step ** np.linspace(-np.ceil(self.hp.scale_num / 2), np.ceil(self.hp.scale_num / 2),
                                                     self.hp.scale_num)

        # cosine window to penalize large displacements
        hann_1d = np.expand_dims(np.hanning(self.final_score_sz), axis=0)
        penalty = np.transpose(hann_1d) * hann_1d
        self.penalty = penalty / np.sum(penalty)

        #Calculate search and target patch sizes
        context = self.design.context * (self.target_w + self.target_h)
        self.z_sz = np.sqrt(np.prod((self.target_w + context) * (self.target_h + context)))
        self.x_sz = float(self.design.search_sz) / self.design.exemplar_sz * self.z_sz

        #Initialize the network
        self.features_x, self.features_z, self.scores, self.z_crops, self.x_crops = self.InitNetwork()

        latest_checkpoint = "/media/engin/63c43c7a-cb63-4c43-b70c-f3cb4d68762a/models/wbaek_colorization/model1_18022020/model.ckpt-56000"

        config1 = tf.ConfigProto()
        config1.gpu_options.visible_device_list = "1"
        config1.gpu_options.per_process_gpu_memory_fraction = 0.45

        #Load the model for search branch
        with self.graph_search.as_default():
            self.session_search = tf.Session(graph=self.graph_search, config=config1)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(self.session_search, latest_checkpoint)

        #Load the model for exemplar branch
        with self.graph_exemplar.as_default():
            self.session_exemplar = tf.Session(graph=self.graph_exemplar, config=config1)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(self.session_exemplar, latest_checkpoint)

        config2 = tf.ConfigProto()
        config2.gpu_options.visible_device_list = "1"
        config2.gpu_options.per_process_gpu_memory_fraction = 0.1
        #Create a session for matching branch
        with self.graph_match.as_default():
            self.session_match = tf.Session(graph=self.graph_match, config=config2)


        #Calculate the score for template
        # Run the template session
        with self.graph_exemplar.as_default():
            self.templates_z_, z_crops_ = self.session_exemplar.run([self.features_z, self.z_crops], feed_dict={
                self.exemplar_ph['filename_ph']: imagepath,
                self.exemplar_ph['pos_x_ph']: self.pos_x,
                self.exemplar_ph['pos_y_ph']: self.pos_y,
                self.exemplar_ph['z_sz_ph']: self.z_sz
            })

        #Write the template image
        z_crops_image = Image.fromarray(np.reshape(z_crops_, (128, 128)).astype(np.uint8))
        z_crops_image.save("/home/engin/Documents/output/template.jpg")

        return

    def reset(self):
        self.session_exemplar.close()
        self.session_search.close()
        self.session_match.close()

    def track(self, imagepath):
        #Calculate the scaled params, scales are calculated in __init__ method.
        scaled_exemplar = self.z_sz * self.scale_factors
        scaled_search_area = self.x_sz * self.scale_factors
        scaled_target_w = self.target_w * self.scale_factors
        scaled_target_h = self.target_h * self.scale_factors

        #Run the search session
        with self.graph_search.as_default():
            features_x_result, x_crops_ = self.session_search.run([self.features_x, self.x_crops], feed_dict={
                self.search_ph['filename_ph']: imagepath,
                self.search_ph['pos_x_ph']: self.pos_x,
                self.search_ph['pos_y_ph']: self.pos_y,
                self.search_ph['x_sz0_ph']: scaled_search_area[0],
                self.search_ph['x_sz1_ph']: scaled_search_area[1],
                self.search_ph['x_sz2_ph']: scaled_search_area[2],
            })

            #Write the images
            x_crops_ = np.reshape(x_crops_, (3,256, 256, 1)).astype(np.uint8)
            x_crops_image_0 = Image.fromarray(np.reshape(x_crops_[0], (256,256)))
            x_crops_image_1 = Image.fromarray(np.reshape(x_crops_[1], (256,256)))
            x_crops_image_2 = Image.fromarray(np.reshape(x_crops_[2], (256, 256)))
            x_crops_image_0.save("/home/engin/Documents/output/x_{}_0.jpg".format(self.track_count))
            x_crops_image_1.save("/home/engin/Documents/output/x_{}_1.jpg".format(self.track_count))
            x_crops_image_2.save("/home/engin/Documents/output/x_{}_2.jpg".format(self.track_count))


        # Run the matching session
        with self.graph_match.as_default():
            scores_ = self.session_match.run(self.scores, feed_dict={
                self.features_x_ph: features_x_result,
                self.features_z_ph: self.templates_z_
            })

        scores_ = np.squeeze(scores_)
        # penalize change of scale
        scores_[0, :, :] = self.hp.scale_penalty * scores_[0, :, :]
        scores_[2, :, :] = self.hp.scale_penalty * scores_[2, :, :]
        # find scale with highest peak (after penalty)
        new_scale_id = np.argmax(np.amax(scores_, axis=(1, 2)))
        # update scaled sizes
        self.x_sz = (1 - self.hp.scale_lr) * self.x_sz + self.hp.scale_lr * scaled_search_area[new_scale_id]
        self.target_w = (1 - self.hp.scale_lr) * self.target_w + self.hp.scale_lr * scaled_target_w[new_scale_id]
        self.target_h = (1 - self.hp.scale_lr) * self.target_h + self.hp.scale_lr * scaled_target_h[new_scale_id]
        # select response with new_scale_id
        score_ = scores_[new_scale_id, :, :]
        score_ = score_ - np.min(score_)
        score_ = score_ / np.sum(score_)
        # apply displacement penalty
        score_ = (1 - self.hp.window_influence) * score_ + self.hp.window_influence * self.penalty
        # Calculate the new center location and confidence
        self.pos_x, self.pos_y, confidence = self._update_target_position(self.pos_x, self.pos_y, score_,
                                                                          self.final_score_sz, self.design.tot_stride,
                                                                          self.design.search_sz, self.hp.response_up,
                                                                          self.x_sz)

        # update the target representation with a rolling average
        if self.hp.z_lr > 0:
            with self.graph_exemplar.as_default():
                new_templates_z_, z_crops_ = self.session_exemplar.run([self.features_z, self.z_crops], feed_dict={
                    self.exemplar_ph['filename_ph']: imagepath,
                    self.exemplar_ph['pos_x_ph']: self.pos_x,
                    self.exemplar_ph['pos_y_ph']: self.pos_y,
                    self.exemplar_ph['z_sz_ph']: self.z_sz
                })

                # Write the template image
                Image.fromarray(np.reshape(z_crops_, (128, 128)).astype(np.uint8)).save("/home/engin/Documents/output/z_{}.jpg".format(self.track_count))

            self.templates_z_ = (1 - self.hp.z_lr) * np.asarray(self.templates_z_) + self.hp.z_lr * np.asarray(
                new_templates_z_)

        # update template patch size
        self.z_sz = (1 - self.hp.scale_lr) * self.z_sz + self.hp.scale_lr * scaled_exemplar[new_scale_id]

        #Increment tracker count
        self.track_count = self.track_count + 1

        # convert <cx,cy,w,h> to <x,y,w,h> and save output
        return vot.Rectangle(self.pos_x - self.target_w / 2, self.pos_y - self.target_h / 2, self.target_w,
                             self.target_h), confidence

    def _update_target_position(self, pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
        # find location of score maximizer
        p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
        # Max value
        confidence = score[p[0]][p[1]]
        # displacement from the center in search area final representation ...
        center = float(final_score_sz - 1) / 2
        disp_in_area = p - center
        # displacement from the center in instance crop
        disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
        # displacement from the center in instance crop (in frame coordinates)
        disp_in_frame = disp_in_xcrop * x_sz / search_sz
        # *position* within frame in frame coordinates
        pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
        return pos_x, pos_y, confidence

    def extract_crops_x(self, im, npad, pos_x, pos_y, sz_src0, sz_src1, sz_src2, sz_dst):
        # take center of the biggest scaled source patch
        c = sz_src2 / 2
        # get top-right corner of bbox and consider padding
        tr_x = npad + tf.cast(tf.round(pos_x - c), tf.int32)
        tr_y = npad + tf.cast(tf.round(pos_y - c), tf.int32)
        # Compute size from rounded co-ords to ensure rectangle lies inside padding.
        width = tf.round(pos_x + c) - tf.round(pos_x - c)
        height = tf.round(pos_y + c) - tf.round(pos_y - c)
        search_area = tf.image.crop_to_bounding_box(im,
                                                    tf.cast(tr_y, tf.int32),
                                                    tf.cast(tr_x, tf.int32),
                                                    tf.cast(height, tf.int32),
                                                    tf.cast(width, tf.int32))
        # TODO: Use computed width and height here?
        offset_s0 = (sz_src2 - sz_src0) / 2
        offset_s1 = (sz_src2 - sz_src1) / 2

        crop_s0 = tf.image.crop_to_bounding_box(search_area,
                                                tf.cast(offset_s0, tf.int32),
                                                tf.cast(offset_s0, tf.int32),
                                                tf.cast(tf.round(sz_src0), tf.int32),
                                                tf.cast(tf.round(sz_src0), tf.int32))
        crop_s0 = tf.image.resize_images(crop_s0, [sz_dst, sz_dst], method=tf.image.ResizeMethod.BILINEAR)
        crop_s1 = tf.image.crop_to_bounding_box(search_area,
                                                tf.cast(offset_s1, tf.int32),
                                                tf.cast(offset_s1, tf.int32),
                                                tf.cast(tf.round(sz_src1), tf.int32),
                                                tf.cast(tf.round(sz_src1), tf.int32))
        crop_s1 = tf.image.resize_images(crop_s1, [sz_dst, sz_dst], method=tf.image.ResizeMethod.BILINEAR)
        crop_s2 = tf.image.resize_images(search_area, [sz_dst, sz_dst], method=tf.image.ResizeMethod.BILINEAR)
        crops = tf.stack([crop_s0, crop_s1, crop_s2])
        return crops

    def open_image(self, imagepath, size):
        #Read the image
        image = cv2.imread(imagepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = np.reshape(cv2.resize(image, size), (size[0], size[1], 1))
        return image

    def extract_crops_z(self, im, npad, pos_x, pos_y, sz_src, sz_dst):
        c = sz_src / 2
        # get top-right corner of bbox and consider padding
        tr_x = npad + tf.cast(tf.round(pos_x - c), tf.int32)
        # Compute size from rounded co-ords to ensure rectangle lies inside padding.
        tr_y = npad + tf.cast(tf.round(pos_y - c), tf.int32)
        width = tf.round(pos_x + c) - tf.round(pos_x - c)
        height = tf.round(pos_y + c) - tf.round(pos_y - c)
        crop = tf.image.crop_to_bounding_box(im,
                                             tf.cast(tr_y, tf.int32),
                                             tf.cast(tr_x, tf.int32),
                                             tf.cast(height, tf.int32),
                                             tf.cast(width, tf.int32))
        crop = tf.image.resize_images(crop, [sz_dst, sz_dst], method=tf.image.ResizeMethod.BILINEAR)
        # crops = tf.stack([crop, crop, crop])
        crops = tf.expand_dims(crop, axis=0)
        return crops

    def pad_frame(self, im, frame_sz, pos_x, pos_y, patch_sz):
        c = patch_sz / 2
        xleft_pad = tf.maximum(0, -tf.cast(tf.round(pos_x - c), tf.int32))
        ytop_pad = tf.maximum(0, -tf.cast(tf.round(pos_y - c), tf.int32))
        xright_pad = tf.maximum(0, tf.cast(tf.round(pos_x + c), tf.int32) - frame_sz[1])
        ybottom_pad = tf.maximum(0, tf.cast(tf.round(pos_y + c), tf.int32) - frame_sz[0])
        npad = tf.reduce_max([xleft_pad, ytop_pad, xright_pad, ybottom_pad])
        paddings = [[npad, npad], [npad, npad], [0, 0]]
        im_padded = im
        # if avg_chan is not None:
        #     im_padded = im_padded - avg_chan
        im_padded = tf.pad(im_padded, paddings, mode='CONSTANT')
        # if avg_chan is not None:
        #     im_padded = im_padded + avg_chan
        return im_padded, npad

    def InitNetwork(self):
        self.graph_search = tf.Graph()
        with self.graph_search.as_default():
            with tf.variable_scope('resnet', reuse=False):
                with tf.name_scope('network') as name_scope:
                    #using default values for batch_norm_decay=0.997, batch_norm_epsilon=1e-5
                    model = ResNetColorizer(is_training=False, data_format='channels_last')

                    #Exemplar placeholders for data and labels
                    #[BATCH, NUM_OF_SAMPLES, H, W, CH]
                    #Input of grayscaled images
                    #Placehholders
                    self.search_ph = {"pos_x_ph": tf.placeholder(tf.float64),
                                      "pos_y_ph": tf.placeholder(tf.float64),
                                      "x_sz0_ph": tf.placeholder(tf.float64),
                                      "x_sz1_ph": tf.placeholder(tf.float64),
                                      "x_sz2_ph": tf.placeholder(tf.float64),
                                      "filename_ph": tf.placeholder(tf.string, [], name='filename')
                                      }
                    #Open the image
                    image_file = tf.read_file(self.search_ph['filename_ph'])
                    # Decode the image as a JPEG file, this will turn it into a Tensor
                    image = tf.image.decode_jpeg(image_file, channels=1)

                    #Get the shape
                    frame_sz = tf.shape(image)

                    # pad with if necessary
                    frame_padded_x, npad_x = self.pad_frame(image, frame_sz,
                                                            self.search_ph['pos_x_ph'], self.search_ph['pos_y_ph'],
                                                            self.search_ph['x_sz2_ph'])
                    frame_padded_x = tf.cast(frame_padded_x, tf.float32)

                    # extract tensor of x_crops (3 scales)
                    x_crops = self.extract_crops_x(frame_padded_x, npad_x,
                                                   self.search_ph['pos_x_ph'], self.search_ph['pos_y_ph'],
                                                   self.search_ph['x_sz0_ph'], self.search_ph['x_sz1_ph'],
                                                   self.search_ph['x_sz2_ph'],
                                                   self.search_sz)

                    x_crops = tf.expand_dims(x_crops, 0)
                    features_x = model.forward(x_crops)

        self.graph_exemplar = tf.Graph()
        with self.graph_exemplar.as_default():
            with tf.variable_scope('resnet', reuse=False):
                with tf.name_scope('network') as name_scope:
                    # using default values for batch_norm_decay=0.997, batch_norm_epsilon=1e-5
                    model = ResNetColorizer(is_training=False, data_format='channels_last')
                    #Template placeholders for data and labels
                    #Input of grayscaled images
                    self.exemplar_ph = {'pos_x_ph': tf.placeholder(tf.float64),
                                        'pos_y_ph': tf.placeholder(tf.float64),
                                        'z_sz_ph': tf.placeholder(tf.float64),
                                        'filename_ph': tf.placeholder(tf.string, [], name='filename')
                                        }

                    #Open the image
                    image_file = tf.read_file(self.exemplar_ph['filename_ph'])
                    # Decode the image as a JPEG file, this will turn it into a Tensor
                    image = tf.image.decode_jpeg(image_file, channels=1)

                    #Get the shape
                    frame_sz = tf.shape(image)

                    #pad with if necessary
                    frame_padded_z, npad_z = self.pad_frame(image, frame_sz,
                                                            self.exemplar_ph['pos_x_ph'], self.exemplar_ph['pos_y_ph'],
                                                            self.exemplar_ph['z_sz_ph'])
                    frame_padded_z = tf.cast(frame_padded_z, tf.float32)

                    # extract tensor of z_crops
                    z_crops = self.extract_crops_z(frame_padded_z, npad_z,
                                                   self.exemplar_ph['pos_x_ph'], self.exemplar_ph['pos_y_ph'],
                                                   self.exemplar_ph['z_sz_ph'],
                                                   self.exemplar_sz)

                    #Feed to the model
                    z_crops = tf.expand_dims(z_crops, 0)
                    features_z = model.forward(z_crops)
                    features_z = tf.squeeze(features_z)
                    features_z = tf.stack([features_z, features_z, features_z])


        self.graph_match = tf.Graph()
        with self.graph_match.as_default():
            self.features_x_ph = tf.placeholder(tf.float32, (3, 32, 32, 64), 'feautes_x_input')
            self.features_z_ph = tf.placeholder(tf.float32, (3, 16, 16, 64), 'features_z_input')
            #match the templates
            # z, x are [B, H, W, C]
            net_z = tf.transpose(self.features_z_ph, perm=[1, 2, 0, 3])
            net_x = tf.transpose(self.features_x_ph, perm=[1, 2, 0, 3])
            # z, x are [H, W, B, C]
            Hz, Wz, B, C = tf.unstack(tf.shape(net_z))
            Hx, Wx, Bx, Cx = tf.unstack(tf.shape(net_x))
            # assert B==Bx, ('Z and X should have same Batch size')
            # assert C==Cx, ('Z and X should have same Channels number')
            net_z = tf.reshape(net_z, (Hz, Wz, B * C, 1))
            net_x = tf.reshape(net_x, (1, Hx, Wx, B * C))
            net_final = tf.nn.depthwise_conv2d(net_x, net_z, strides=[1, 1, 1, 1], padding='VALID')
            # final is [1, Hf, Wf, BC]
            net_final = tf.concat(tf.split(net_final, 3, axis=3), axis=0)
            # final is [B, Hf, Wf, C]
            scores = tf.expand_dims(tf.reduce_sum(net_final, axis=3), axis=3)
            scores_up = tf.image.resize_images(scores, [self.final_score_sz, self.final_score_sz],
                                               method=tf.image.ResizeMethod.BICUBIC, align_corners=True)


        return features_x, features_z, scores_up, z_crops, x_crops
