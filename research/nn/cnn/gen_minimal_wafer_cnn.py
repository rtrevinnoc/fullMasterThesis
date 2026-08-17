import sys
import os

plot_nn_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../PlotNeuralNet"))
sys.path.append(plot_nn_dir)

from pycore.tikzeng import *

arch = [
    to_head("../../../PlotNeuralNet"),
    to_cor(),
    to_begin(),
    
    # Input Map (white / light pink)
    to_Conv("input", s_filer=64, n_filer=1, offset="(0,0,0)", to="(0,0,0)", height=32, depth=32, width=1.0, caption="\\textbf{Map}"),
    
    # C1+Pool (yellow)
    to_Conv("c1_pool", s_filer=32, n_filer=16, offset="(2.2,0,0)", to="(input-east)", height=22, depth=22, width=2.0, caption="\\textbf{C1+Pool}"),
    to_connection("input", "c1_pool"),
    
    # C2+Pool (yellow)
    to_Conv("c2_pool", s_filer=16, n_filer=32, offset="(2.2,0,0)", to="(c1_pool-east)", height=14, depth=14, width=3.0, caption="\\textbf{C2+Pool}"),
    to_connection("c1_pool", "c2_pool"),
    
    # C3+Pool (yellow)
    to_Conv("c3_pool", s_filer=8, n_filer=64, offset="(2.2,0,0)", to="(c2_pool-east)", height=8, depth=8, width=4.2, caption="\\textbf{C3+Pool}"),
    to_connection("c2_pool", "c3_pool"),
    
    # FC (purple / lavender)
    to_Conv("fc", s_filer=128, n_filer="", offset="(2.6,0,0)", to="(c3_pool-east)", height=3, depth=26, width=1.8, caption="\\textbf{FC}"),
    to_connection("c3_pool", "fc"),
    
    # Softmax (purple / lavender)
    to_SoftMax("softmax", s_filer=9, offset="(2.2,0,0)", to="(fc-east)", height=3, depth=8, width=1.8, opacity=0.8, caption="\\textbf{Softmax}"),
    to_connection("fc", "softmax"),
    
    to_end()
]

def main():
    outfile = os.path.join(os.path.dirname(__file__), "wafer_cnn_minimal.tex")
    to_generate(arch, outfile)
    print(f"Generated minimal PlotNeuralNet TikZ at {outfile}")

if __name__ == "__main__":
    main()
