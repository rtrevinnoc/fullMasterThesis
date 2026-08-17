import sys
import os

plot_nn_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../PlotNeuralNet"))
sys.path.append(plot_nn_dir)

from pycore.tikzeng import *

arch = [
    to_head("../../../PlotNeuralNet"),
    to_cor(),
    to_begin(),
    
    # Input
    to_Conv("input", s_filer=64, n_filer=1, offset="(0,0,0)", to="(0,0,0)", height=30, depth=30, width=1, caption="Input\\\\$64{\\times}64{\\times}1$"),
    
    # Conv1 + ReLU
    to_ConvConvRelu("conv1", s_filer=64, n_filer=(16, 16), offset="(1.8,0,0)", to="(input-east)", height=30, depth=30, width=(1.5, 1.5), caption="Conv1 + ReLU\\\\$16\\text{ f, }3{\\times}3$"),
    to_connection("input", "conv1"),
    
    # Pool1
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)", height=20, depth=20, width=1.5, opacity=0.5, caption="Pool1\\\\$32{\\times}32{\\times}16$"),
    
    # Conv2 + ReLU
    to_ConvConvRelu("conv2", s_filer=32, n_filer=(32, 32), offset="(1.8,0,0)", to="(pool1-east)", height=20, depth=20, width=(2.2, 2.2), caption="Conv2 + ReLU\\\\$32\\text{ f, }3{\\times}3$"),
    to_connection("pool1", "conv2"),
    
    # Pool2
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=13, depth=13, width=2.2, opacity=0.5, caption="Pool2\\\\$16{\\times}16{\\times}32$"),
    
    # Conv3 + ReLU
    to_ConvConvRelu("conv3", s_filer=16, n_filer=(64, 64), offset="(1.8,0,0)", to="(pool2-east)", height=13, depth=13, width=(3.2, 3.2), caption="Conv3 + ReLU\\\\$64\\text{ f, }3{\\times}3$"),
    to_connection("pool2", "conv3"),
    
    # Pool3
    to_Pool("pool3", offset="(0,0,0)", to="(conv3-east)", height=7, depth=7, width=3.2, opacity=0.5, caption="Pool3\\\\$8{\\times}8{\\times}64$"),
    
    # FC1
    to_Conv("fc1", s_filer=128, n_filer=1, offset="(2.2,0,0)", to="(pool3-east)", height=3, depth=26, width=3, caption="FC1 + Drop\\\\$128\\text{ units}$"),
    to_connection("pool3", "fc1"),
    
    # Output FC2
    to_SoftMax("fc2", s_filer=9, offset="(2.0,0,0)", to="(fc1-east)", height=3, depth=8, width=2.5, opacity=0.8, caption="Output\\\\$9\\text{ classes}$"),
    to_connection("fc1", "fc2"),
    
    to_end()
]

def main():
    outfile = os.path.join(os.path.dirname(__file__), "wafer_cnn_plotneuralnet.tex")
    to_generate(arch, outfile)
    print(f"Generated PlotNeuralNet TikZ at {outfile}")

if __name__ == "__main__":
    main()
