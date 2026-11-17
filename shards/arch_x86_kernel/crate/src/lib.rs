#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(asm_goto_with_outputs)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "head64.rs"]
mod __slate_unit_head64;
#[path = "ebda.rs"]
mod __slate_unit_ebda;
#[path = "platform_quirks.rs"]
mod __slate_unit_platform_quirks;
#[path = "process_64.rs"]
mod __slate_unit_process_64;
#[path = "signal.rs"]
mod __slate_unit_signal;
#[path = "signal_64.rs"]
mod __slate_unit_signal_64;
#[path = "traps.rs"]
mod __slate_unit_traps;
#[path = "idt.rs"]
mod __slate_unit_idt;
#[path = "irq.rs"]
mod __slate_unit_irq;
#[path = "irq_64.rs"]
mod __slate_unit_irq_64;
#[path = "dumpstack_64.rs"]
mod __slate_unit_dumpstack_64;
#[path = "time.rs"]
mod __slate_unit_time;
#[path = "ioport.rs"]
mod __slate_unit_ioport;
#[path = "dumpstack.rs"]
mod __slate_unit_dumpstack;
#[path = "nmi.rs"]
mod __slate_unit_nmi;
#[path = "ldt.rs"]
mod __slate_unit_ldt;
#[path = "setup.rs"]
mod __slate_unit_setup;
#[path = "x86_init.rs"]
mod __slate_unit_x86_init;
#[path = "i8259.rs"]
mod __slate_unit_i8259;
#[path = "irqinit.rs"]
mod __slate_unit_irqinit;
#[path = "jump_label.rs"]
mod __slate_unit_jump_label;
#[path = "irq_work.rs"]
mod __slate_unit_irq_work;
#[path = "probe_roms.rs"]
mod __slate_unit_probe_roms;
#[path = "sys_ia32.rs"]
mod __slate_unit_sys_ia32;
#[path = "signal_32.rs"]
mod __slate_unit_signal_32;
#[path = "sys_x86_64.rs"]
mod __slate_unit_sys_x86_64;
#[path = "espfix_64.rs"]
mod __slate_unit_espfix_64;
#[path = "ksysfs.rs"]
mod __slate_unit_ksysfs;
#[path = "bootflag.rs"]
mod __slate_unit_bootflag;
#[path = "e820.rs"]
mod __slate_unit_e820;
#[path = "pci_dma.rs"]
mod __slate_unit_pci_dma;
#[path = "quirks.rs"]
mod __slate_unit_quirks;
#[path = "kdebugfs.rs"]
mod __slate_unit_kdebugfs;
#[path = "alternative.rs"]
mod __slate_unit_alternative;
#[path = "i8253.rs"]
mod __slate_unit_i8253;
#[path = "hw_breakpoint.rs"]
mod __slate_unit_hw_breakpoint;
#[path = "tsc.rs"]
mod __slate_unit_tsc;
#[path = "tsc_msr.rs"]
mod __slate_unit_tsc_msr;
#[path = "io_delay.rs"]
mod __slate_unit_io_delay;
#[path = "rtc.rs"]
mod __slate_unit_rtc;
#[path = "resource.rs"]
mod __slate_unit_resource;
#[path = "static_call.rs"]
mod __slate_unit_static_call;
#[path = "process.rs"]
mod __slate_unit_process;
#[path = "ptrace.rs"]
mod __slate_unit_ptrace;
#[path = "tls.rs"]
mod __slate_unit_tls;
#[path = "step.rs"]
mod __slate_unit_step;
#[path = "i8237.rs"]
mod __slate_unit_i8237;
#[path = "stacktrace.rs"]
mod __slate_unit_stacktrace;
#[path = "reboot.rs"]
mod __slate_unit_reboot;
#[path = "msr.rs"]
mod __slate_unit_msr;
#[path = "cpuid.rs"]
mod __slate_unit_cpuid;
#[path = "early_quirks.rs"]
mod __slate_unit_early_quirks;
#[path = "smp.rs"]
mod __slate_unit_smp;
#[path = "smpboot.rs"]
mod __slate_unit_smpboot;
#[path = "tsc_sync.rs"]
mod __slate_unit_tsc_sync;
#[path = "setup_percpu.rs"]
mod __slate_unit_setup_percpu;
#[path = "mpparse.rs"]
mod __slate_unit_mpparse;
#[path = "trace_clock.rs"]
mod __slate_unit_trace_clock;
#[path = "trace.rs"]
mod __slate_unit_trace;
#[path = "rethook.rs"]
mod __slate_unit_rethook;
#[path = "vmcore_info_64.rs"]
mod __slate_unit_vmcore_info_64;
#[path = "machine_kexec_64.rs"]
mod __slate_unit_machine_kexec_64;
#[path = "crash_dump_64.rs"]
mod __slate_unit_crash_dump_64;
#[path = "crash.rs"]
mod __slate_unit_crash;
#[path = "module.rs"]
mod __slate_unit_module;
#[path = "early_printk.rs"]
mod __slate_unit_early_printk;
#[path = "hpet.rs"]
mod __slate_unit_hpet;
#[path = "amd_nb.rs"]
mod __slate_unit_amd_nb;
#[path = "amd_node.rs"]
mod __slate_unit_amd_node;
#[path = "kvm.rs"]
mod __slate_unit_kvm;
#[path = "kvmclock.rs"]
mod __slate_unit_kvmclock;
#[path = "paravirt.rs"]
mod __slate_unit_paravirt;
#[path = "paravirt_spinlocks.rs"]
mod __slate_unit_paravirt_spinlocks;
#[path = "pvclock.rs"]
mod __slate_unit_pvclock;
#[path = "pcspeaker.rs"]
mod __slate_unit_pcspeaker;
#[path = "check.rs"]
mod __slate_unit_check;
#[path = "uprobes.rs"]
mod __slate_unit_uprobes;
#[path = "perf_regs.rs"]
mod __slate_unit_perf_regs;
#[path = "itmt.rs"]
mod __slate_unit_itmt;
#[path = "umip.rs"]
mod __slate_unit_umip;
#[path = "unwind_orc.rs"]
mod __slate_unit_unwind_orc;
#[path = "callthunks.rs"]
mod __slate_unit_callthunks;
#[path = "cet.rs"]
mod __slate_unit_cet;
#[path = "audit_64.rs"]
mod __slate_unit_audit_64;
#[path = "mmconf_fam10h_64.rs"]
mod __slate_unit_mmconf_fam10h_64;
#[path = "vsmp_64.rs"]
mod __slate_unit_vsmp_64;

mod __slate_panic {
    unsafe extern "C" {
        fn panic(fmt: *const core::ffi::c_char, ...) -> !;
    }

    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        unsafe { panic(c"rust panic\n".as_ptr()) }
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

