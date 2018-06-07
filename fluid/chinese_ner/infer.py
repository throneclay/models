import numpy as np
import paddle.fluid as fluid
import paddle

import reader


def load_reverse_dict(dict_path):
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def infer(model_path, batch_size, test_data_file, target_file):
    word = fluid.layers.data(name='word', shape=[1], dtype='int64', lod_level=1)
    mention = fluid.layers.data(
        name='mention', shape=[1], dtype='int64', lod_level=1)
    target = fluid.layers.data(
        name='target', shape=[1], dtype='int64', lod_level=1)

    label_reverse_dict = load_reverse_dict(target_file)

    test_data = paddle.batch(
        reader.file_reader(test_data_file), batch_size=batch_size)
    place = fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[word, mention], place=place)
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(model_path, exe)
        print_res = ""
        for data in test_data():
            crf_decode = exe.run(inference_program,
                                 feed=feeder.feed(data),
                                 fetch_list=fetch_targets,
                                 return_numpy=False)
            lod_info = (crf_decode[0].lod())[0]
            np_data = np.array(crf_decode[0])
            assert len(data) == len(lod_info) - 1
            for sen_index in xrange(len(data)):
                assert len(data[sen_index][0]) == lod_info[
                    sen_index + 1] - lod_info[sen_index]
                word_index = 0
                for tag_index in xrange(lod_info[sen_index],
                                        lod_info[sen_index + 1]):
                    word = str(data[sen_index][0][word_index])
                    #gold_tag = label_reverse_dict[data[sen_index][2][
                    #    word_index]]
                    #tag = label_reverse_dict[np_data[tag_index][0]]
                    print_res += str(np_data[tag_index][0]) +" "
                    word_index += 1
            print_res +="\n"

        print print_res
if __name__ == "__main__":
    infer(
        model_path="/home/public/chinese_ner/params_pass_0",
        #model_path="model/params_batch_450000",
        batch_size=6,
        test_data_file="data/test_files",
        target_file="data/label_dict")
