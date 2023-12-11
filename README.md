# FlightLLM Test Demo

This demo is for testing FlightLLM implementation on the Xilinx Alveo U280 FPGA.
Our submission can be divided into two parts.

1. Performance profile (see `profile/README.md` for details): It is used to compare the performance of the GPU baseline with the simulation performance of the VHK158 FGPA and calculate the speedup ratio, which can be compared with Figure 12 in the paper. The performance results of the VHK158 are based on the software run and no hardware is involved. 

2. FPGA on-board testing (see `fpga_implementation/README.md` for details): It is a hardware on-board test on the U280 FPGA to measure the correctness and performance of the paper design.
