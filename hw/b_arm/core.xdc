# Reference only: synth.tcl applies these after synthesis, with configurable period.
create_clock -name core_clk -period 10.000 [get_ports clk]
set_clock_uncertainty 0.200 [get_clocks core_clk]
set_input_delay 1.000 -clock core_clk [get_ports {rst start in_valid out_ready in_code[*]}]
set_output_delay 1.000 -clock core_clk [get_ports {in_ready out_valid out_bit busy done status[*] rounds[*] total_cycles[*]}]
