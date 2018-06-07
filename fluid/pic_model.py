import paddle.fluid as fluid
import os
from paddle.fluid import debuger

if __name__ == "__main__":

    #ModelPath = '/home/zhangshuai20/workspace/paddle_models/models/fluid/sequence_tagging_for_ner/models/params_pass_0'
    ModelPath = '/home/zhangshuai20/workspace/paddle_models/models_master/fluid/chinese_ner/output/params_pass_0'
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    scope = fluid.core.Scope()

    with fluid.scope_guard(scope):
        if os.path.exists(ModelPath + 'model') and os.path.exists(ModelPath + 'params'):
            [net_program, feed_target_names, fetch_targets] = \
            fluid.io.load_inference_model(ModelPath, exe, 'model', 'params')
        else:
            [net_program, feed_target_names, fetch_targets] = \
            fluid.io.load_inference_model(ModelPath, exe)
        global_block = net_program.global_block()
        debuger.draw_block_graphviz(global_block, path="./map.dot")
