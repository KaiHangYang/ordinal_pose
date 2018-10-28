import os
import sys
import tensorflow as tf
import numpy as np

def get_data_iterator(img_list, lbl_list, batch_size, name="", is_shuffle=True):
    data_sum = len(img_list)
    print("DataSet: {}. Sum: {}. BatchSize: {}. Shuffle: {}".format(name, data_sum, batch_size, is_shuffle))

    with tf.name_scope(name):
        assert(len(img_list) == len(lbl_list))
        for i in range(len(img_list)):
            assert(os.path.basename(img_list[i]).split(".")[0] == os.path.basename(lbl_list[i]).split(".")[0])

        img_dataset = tf.data.Dataset.from_tensor_slices(img_list)
        lbl_dataset = tf.data.Dataset.from_tensor_slices(lbl_list)

        if is_shuffle:
            dataset = tf.data.Dataset.zip((img_dataset, lbl_dataset)).shuffle(data_sum).repeat().batch(batch_size)
        else:
            dataset = tf.data.Dataset.zip((img_dataset, lbl_dataset)).repeat().batch(batch_size)

        iterator = tf.data.Iterator.from_structure(output_types=dataset.output_types, output_shapes=dataset.output_shapes)

        return iterator.get_next(), iterator.make_initializer(dataset)

if __name__ == "__main__":
    iteration = 100000
    test_count = 10
    img_list = [str(i) for i in range(test_count)]
    lbl_list = ["/usr/local/" + str(i) for i in range(test_count)]

    iter, dataset_init_op = get_data_iterator(img_list, lbl_list, 4)

    counter_arr = np.zeros(test_count)
    with tf.Session() as sess:
        sess.run(dataset_init_op)

        for i in range(iteration):
            cur_data, cur_label = sess.run(iter)
            print(cur_data)
            for j in cur_data:
                counter_arr[int(j)] += 1

    print(counter_arr)
