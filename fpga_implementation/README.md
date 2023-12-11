# FlightLLM Test Demo

This demo is for testing FlightLLM implementation on the Xilinx Alveo U280 FPGA.

## Runtime Environment

Before running this demo, ensure that your system meets the following runtime environment requirements.

### Operating System
```plaintext
1. Ubuntu 20.04 LTS
2. C++ Runtime
```
### Hardware

This demo requires a computer with a U280. Ensure that your system has a U280 meeting the following environment:
```plaintext
1. Xilinx Platforms xilinx_u280_gen3x16_xdma_1_202211_1
2. Xilinx XRT Version 2.15.225, Branch 2023.1
```

## Directory Structure
```plaintext
├── host                          
│    └── fpgaHost                   # demo executable file
├── bitstream                          
│    └── stc-v1.xclbin              # u280 xclbin file
├── case                            
│   └── token_64_single_layer    
│       ├── info.yaml               # case info    
│       ├── input                   # test input files
│       ├── inst                    # instruction files
│       ├── output                  # golden files
│       └── param                   # weights files
└── README.md
```

## Usage
```plaintext
cd fpga_implementation
host/fpgaHost <xclbin_path> <case_path>

For example:
host/fpgaHost bitstream/stc-v1.xclbin case/token_64_single_layer

In this example:
bitstream/stc.xclbin is a hardware implementation of FlightLLM.
case/token_64_single_layer contains instruction files, input files and golden files. 

After the demo completes its run, it will compare output with the golden file:

[DEBUG] [2023-12-09 16:36:40] COMPARE WITH case/token_64_single_layer/output/output.ch00.rtl.bin, TEST OK
[DEBUG] [2023-12-09 16:36:40] COMPARE WITH case/token_64_single_layer/output/output.ch04.rtl.bin, TEST OK
[DEBUG] [2023-12-09 16:36:40] COMPARE WITH case/token_64_single_layer/output/output.ch08.rtl.bin, TEST OK
[DEBUG] [2023-12-09 16:36:40] COMPARE WITH case/token_64_single_layer/output/output.ch12.rtl.bin, TEST OK
[DEBUG] [2023-12-09 16:36:40] COMPARE WITH case/token_64_single_layer/output/output.ch16.rtl.bin, TEST OK
[DEBUG] [2023-12-09 16:36:40] COMPARE WITH case/token_64_single_layer/output/output.ch20.rtl.bin, TEST OK
[DEBUG] [2023-12-09 16:36:40] COMPARE WITH case/token_64_single_layer/output/output.ch24.rtl.bin, TEST OK
[DEBUG] [2023-12-09 16:36:40] COMPARE WITH case/token_64_single_layer/output/output.ch28.rtl.bin, TEST OK
```