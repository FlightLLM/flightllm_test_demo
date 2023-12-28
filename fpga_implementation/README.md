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
│   └── decode_token_128_ae                        
│   └── decode_token_512_ae    
│       ├── info.yaml               # case info    
│       ├── input                   # test input files
│       ├── inst                    # instruction files
│       ├── output                  # golden files
│       └── param                   # weights files
└── README.md
```

## Usage
```plaintext
cd fpga_implementation  && chmod +x host/fpgaHost
host/fpgaHost <xclbin_path> <case_path>

For example:
host/fpgaHost bitstream/stc-v1.xclbin case/decode_token_512_ae

In this example:
bitstream/stc.xclbin is a hardware implementation of FlightLLM.
case/decode_token_512_ae contains instruction files, input files and golden files. 

After the demo completes its run, it will compare output with the golden file:

[DEBUG] [2023-12-25 15:55:58] ./output_0.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch00.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_1.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch01.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_2.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch02.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_3.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch04.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_4.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch05.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_5.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch06.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_6.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch08.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_7.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch09.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_8.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch10.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_9.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch12.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_10.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch13.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_11.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch14.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_12.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch15.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_13.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch16.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:58] ./output_14.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch17.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:59] ./output_15.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch18.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:59] ./output_16.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch20.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:59] ./output_17.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch21.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:59] ./output_18.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch22.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:59] ./output_19.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch24.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:59] ./output_20.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch25.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:59] ./output_21.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch26.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:59] ./output_22.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch28.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:59] ./output_23.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch29.rtl.bin, TEST OK
[DEBUG] [2023-12-25 15:55:59] ./output_24.bin COMPARE WITH /home/fpga5/case_isa_v1.3/decode_token_128_ae/output/output.ch30.rtl.bin, TEST OK
```