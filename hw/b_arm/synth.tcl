# Dedicated tool host only. No pin constraints, bitstream, or board claims.
set_param general.maxThreads 1
if {$argc < 2} {error "usage: synth.tcl generated_directory output_directory ?period_ns? ?synth_only?"}
set src [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set period 10.0
if {$argc >= 3} {set period [lindex $argv 2]}
set synth_only 0
if {$argc >= 4} {set synth_only [lindex $argv 3]}
set part xc7vx415tffg1157-1
if {[llength [get_parts -quiet $part]] != 1} {error "reference part unavailable"}
file mkdir $out
set begin [clock seconds]
read_verilog -sv [file join $src core.sv]
synth_design -top core -part $part -mode out_of_context
create_clock -name core_clk -period $period [get_ports clk]
set_clock_uncertainty 0.200 [get_clocks core_clk]
set_input_delay 1.000 -clock core_clk [get_ports {rst start in_valid out_ready in_code[*]}]
set_output_delay 1.000 -clock core_clk [get_ports {in_ready out_valid out_bit busy done status[*] rounds[*] total_cycles[*]}]
write_xdc -force [file join $out applied.xdc]
report_utilization -file [file join $out utilization_synth.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -file [file join $out timing_synth.rpt]
if {!$synth_only} {
 opt_design
 place_design
 phys_opt_design
 route_design
 report_utilization -hierarchical -file [file join $out utilization_route.rpt]
 report_timing_summary -delay_type min_max -report_unconstrained -file [file join $out timing_route.rpt]
 report_drc -file [file join $out drc_route.rpt]
 report_route_status -file [file join $out route_status.rpt]
 check_timing -verbose -file [file join $out check_timing.rpt]
 write_checkpoint -force [file join $out routed.dcp]
}
set f [open [file join $out metadata.txt] w]
puts $f "tool=[version -short]\npart=$part\nperiod_ns=$period\nuncertainty_ns=0.2\ninterface_delay_ns=1.0\nsynth_only=$synth_only\nelapsed_seconds=[expr {[clock seconds]-$begin}]"
close $f
file copy -force [file join $src manifest.json] [file join $out manifest.json]
