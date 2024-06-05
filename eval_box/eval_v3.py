import os, sys
from thop import profile
os.chdir("/taskroot/scripts") # path/to/scripts
sys.path.append(".")
from interface import init
from ipdb import set_trace

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    input, net, inference = init()
    datasets = os.listdir("../data/")
    datasets.sort()
    for dataset in datasets:
        protocols = os.listdir("../data/" + dataset + "/index")
        output_dir = '../outputs/'+dataset
        print(dataset)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        open(os.path.join(output_dir, 'test'), 'w').close()
        for i, protocol in enumerate(protocols):
            protocol_path = "../data/" + dataset + "/index/" + protocol

            output_path = os.path.join(output_dir, protocol.split('.')[0], 'eval_result.txt')   # output pred
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))

            inference(protocol_path, output_path)
        open(os.path.join(output_dir, 'success'), 'w').close()   # write success
    if isinstance(net, list):
        flops = params = 0
        for i in range(len(net)):
            flop, param = profile(net[i], inputs=(input[i],))
            flops += flop
            params += param
    elif isinstance(input, list):
        flops, params = profile(net, inputs=(tuple(input)))
    else:
        flops, params = profile(net, inputs=(input,))
    with open('../outputs/model.txt', 'w') as f:
        f.write(str(int(flops)) + ',' + str(int(params)))
