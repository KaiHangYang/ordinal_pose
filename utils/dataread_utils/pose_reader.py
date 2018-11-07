import os
import sys
import tensorflow as tf
import numpy as np

#### The second part the network only need the lbl
#### The input image will be painted according the lbl

def get_data_iterator(lbl_list, batch_size, name="", is_shuffle=True):
    data_sum = len(lbl_list)
    print("DataSet: {}. Sum: {}. BatchSize: {}. Shuffle: {}".format(name, data_sum, batch_size, is_shuffle))

    with tf.name_scope(name):
        lbl_dataset = tf.data.Dataset.from_tensor_slices(lbl_list)

        if is_shuffle:
            dataset = lbl_dataset.shuffle(data_sum).repeat().batch(batch_size)
        else:
            dataset = lbl_dataset.repeat().batch(batch_size)

        iterator = tf.data.Iterator.from_structure(output_types=dataset.output_types, output_shapes=dataset.output_shapes)

        return iterator.get_next(), iterator.make_initializer(dataset)

if __name__ == "__main__":
    iteration = 100000
    test_count = 10
    lbl_list = ["/usr/local/" + str(i) for i in range(test_count)]

    iter, dataset_init_op = get_data_iterator(lbl_list, 4)

    counter_arr = np.zeros(test_count)
    with tf.Session() as sess:
        sess.run(dataset_init_op)

        for i in range(iteration):
            cur_label = sess.run(iter)
            for j in cur_label:
                counter_arr[int(os.path.basename(j))] += 1

    print(counter_arr)
