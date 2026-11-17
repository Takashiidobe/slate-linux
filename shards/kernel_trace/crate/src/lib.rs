#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "trace_clock.rs"]
mod __slate_unit_trace_clock;
#[path = "ring_buffer.rs"]
mod __slate_unit_ring_buffer;
#[path = "trace.rs"]
mod __slate_unit_trace;
#[path = "trace_output.rs"]
mod __slate_unit_trace_output;
#[path = "trace_seq.rs"]
mod __slate_unit_trace_seq;
#[path = "trace_stat.rs"]
mod __slate_unit_trace_stat;
#[path = "trace_printk.rs"]
mod __slate_unit_trace_printk;
#[path = "trace_pid.rs"]
mod __slate_unit_trace_pid;
#[path = "pid_list.rs"]
mod __slate_unit_pid_list;
#[path = "trace_sched_switch.rs"]
mod __slate_unit_trace_sched_switch;
#[path = "trace_nop.rs"]
mod __slate_unit_trace_nop;
#[path = "blktrace.rs"]
mod __slate_unit_blktrace;
#[path = "trace_events.rs"]
mod __slate_unit_trace_events;
#[path = "trace_export.rs"]
mod __slate_unit_trace_export;
#[path = "trace_event_perf.rs"]
mod __slate_unit_trace_event_perf;
#[path = "trace_events_filter.rs"]
mod __slate_unit_trace_events_filter;
#[path = "trace_events_trigger.rs"]
mod __slate_unit_trace_events_trigger;
#[path = "trace_eprobe.rs"]
mod __slate_unit_trace_eprobe;
#[path = "trace_kprobe.rs"]
mod __slate_unit_trace_kprobe;
#[path = "error_report_traces.rs"]
mod __slate_unit_error_report_traces;
#[path = "power_traces.rs"]
mod __slate_unit_power_traces;
#[path = "rpm_traces.rs"]
mod __slate_unit_rpm_traces;
#[path = "trace_dynevent.rs"]
mod __slate_unit_trace_dynevent;
#[path = "trace_probe.rs"]
mod __slate_unit_trace_probe;
#[path = "trace_uprobe.rs"]
mod __slate_unit_trace_uprobe;
#[path = "rethook.rs"]
mod __slate_unit_rethook;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

