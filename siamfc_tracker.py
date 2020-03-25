import src.siamese as siam
from src.parse_arguments import parse_arguments
import numpy as np
import tensorflow as tf
import vot

class SiamFCTracker:
    def __init__(self, image_path, region):
        #Parse the arguments
        self.hp, self.evaluation, self.run, self.env, self.design = parse_arguments()

        #Get first frame image and ground-truth
        self.region = region
        self.pos_x = region.x + region.width / 2
        self.pos_y = region.y + region.height / 2
        self.target_w = region.width
        self.target_h = region.height
        self.bbox = self.pos_x - self.target_w/2, self.pos_y - self.target_h/2, self.target_w, self.target_h

        #Calculate the size of final score (upscaled size of score matrix, where score matrix
        # is convolution of results of two branches of siamese network)
        self.final_score_sz = self.hp.response_up * (self.design.score_sz - 1) + 1

        #Initialize the network and load the weights
        self.filename, self.image, self.templates_z, \
        self.templates_x, self.scores, self.scores_original = siam.build_tracking_graph(self.final_score_sz, self.design, self.env)

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

        #Create a tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = "1"
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(config=config)
        with self.sess.as_default():
            tf.global_variables_initializer().run()
            # Coordinate the loading of image files.
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(coord=self.coord)

            self.run_opts = {}

            #Calculate the template for the given region
            image_, self.templates_z_ = self.sess.run([self.image, self.templates_z], feed_dict={
                                                                            siam.pos_x_ph: self.pos_x,
                                                                            siam.pos_y_ph: self.pos_y,
                                                                            siam.z_sz_ph: self.z_sz,
                                                                            self.filename: image_path})

        return


    def track(self, image_path):
        #Calculate the scaled params, scales are calculated in __init__ method.
        scaled_exemplar = self.z_sz * self.scale_factors
        scaled_search_area = self.x_sz * self.scale_factors
        scaled_target_w = self.target_w * self.scale_factors
        scaled_target_h = self.target_h * self.scale_factors

        #Run the network
        with self.sess.as_default():
            image_, scores_, scores_original_, self.templates_x_, self.templates_z_ = self.sess.run(
                [self.image, self.scores, self.scores_original, self.templates_x, self.templates_z],
                feed_dict={
                    siam.pos_x_ph: self.pos_x,
                    siam.pos_y_ph: self.pos_y,
                    siam.x_sz0_ph: scaled_search_area[0],
                    siam.x_sz1_ph: scaled_search_area[1],
                    siam.x_sz2_ph: scaled_search_area[2],
                    self.templates_z: np.squeeze(self.templates_z_),
                    self.filename: image_path,
                }, **self.run_opts)
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
            #Calculate the new center location and confidence
            self.pos_x, self.pos_y, confidence = self._update_target_position(self.pos_x, self.pos_y, score_, self.final_score_sz, self.design.tot_stride,
                                                   self.design.search_sz, self.hp.response_up, self.x_sz)

            # update the target representation with a rolling average
            if self.hp.z_lr > 0:
                new_templates_z_ = self.sess.run([self.templates_z], feed_dict={
                    siam.pos_x_ph: self.pos_x,
                    siam.pos_y_ph: self.pos_y,
                    siam.z_sz_ph: self.z_sz,
                    self.image: image_
                })

                self.templates_z_ = (1 - self.hp.z_lr) * np.asarray(self.templates_z_) + self.hp.z_lr * np.asarray(new_templates_z_)

            # update template patch size
            self.z_sz = (1 - self.hp.scale_lr) * self.z_sz + self.hp.scale_lr * scaled_exemplar[new_scale_id]

            # convert <cx,cy,w,h> to <x,y,w,h> and save output
            return vot.Rectangle(self.pos_x - self.target_w / 2, self.pos_y - self.target_h / 2, self.target_w, self.target_h), confidence

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

