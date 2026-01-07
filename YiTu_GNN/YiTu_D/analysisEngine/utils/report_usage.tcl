open_project [lindex $argv 0]
open_run  impl_1
report_utilization -file  util_full.report
report_utilization -hierarchical -hierarchical_depth 3 -file  util_slr.report
report_power -file power.report