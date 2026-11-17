#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(asm_goto_with_outputs)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "fork.rs"]
mod __slate_unit_fork;
#[path = "exec_domain.rs"]
mod __slate_unit_exec_domain;
#[path = "exec_state.rs"]
mod __slate_unit_exec_state;
#[path = "panic.rs"]
mod __slate_unit_panic;
#[path = "cpu.rs"]
mod __slate_unit_cpu;
#[path = "exit.rs"]
mod __slate_unit_exit;
#[path = "softirq.rs"]
mod __slate_unit_softirq;
#[path = "resource.rs"]
mod __slate_unit_resource;
#[path = "sysctl.rs"]
mod __slate_unit_sysctl;
#[path = "capability.rs"]
mod __slate_unit_capability;
#[path = "ptrace.rs"]
mod __slate_unit_ptrace;
#[path = "user.rs"]
mod __slate_unit_user;
#[path = "signal.rs"]
mod __slate_unit_signal;
#[path = "sys.rs"]
mod __slate_unit_sys;
#[path = "umh.rs"]
mod __slate_unit_umh;
#[path = "workqueue.rs"]
mod __slate_unit_workqueue;
#[path = "pid.rs"]
mod __slate_unit_pid;
#[path = "task_work.rs"]
mod __slate_unit_task_work;
#[path = "extable.rs"]
mod __slate_unit_extable;
#[path = "params.rs"]
mod __slate_unit_params;
#[path = "kthread.rs"]
mod __slate_unit_kthread;
#[path = "sys_ni.rs"]
mod __slate_unit_sys_ni;
#[path = "nsproxy.rs"]
mod __slate_unit_nsproxy;
#[path = "nstree.rs"]
mod __slate_unit_nstree;
#[path = "nscommon.rs"]
mod __slate_unit_nscommon;
#[path = "notifier.rs"]
mod __slate_unit_notifier;
#[path = "ksysfs.rs"]
mod __slate_unit_ksysfs;
#[path = "cred.rs"]
mod __slate_unit_cred;
#[path = "reboot.rs"]
mod __slate_unit_reboot;
#[path = "async.rs"]
mod __slate_unit_async;
#[path = "range.rs"]
mod __slate_unit_range;
#[path = "smpboot.rs"]
mod __slate_unit_smpboot;
#[path = "ucount.rs"]
mod __slate_unit_ucount;
#[path = "regset.rs"]
mod __slate_unit_regset;
#[path = "ksyms_common.rs"]
mod __slate_unit_ksyms_common;
#[path = "groups.rs"]
mod __slate_unit_groups;
#[path = "kcmp.rs"]
mod __slate_unit_kcmp;
#[path = "freezer.rs"]
mod __slate_unit_freezer;
#[path = "profile.rs"]
mod __slate_unit_profile;
#[path = "stacktrace.rs"]
mod __slate_unit_stacktrace;
#[path = "dma.rs"]
mod __slate_unit_dma;
#[path = "smp.rs"]
mod __slate_unit_smp;
#[path = "uid16.rs"]
mod __slate_unit_uid16;
#[path = "kallsyms.rs"]
mod __slate_unit_kallsyms;
#[path = "acct.rs"]
mod __slate_unit_acct;
#[path = "vmcore_info.rs"]
mod __slate_unit_vmcore_info;
#[path = "elfcorehdr.rs"]
mod __slate_unit_elfcorehdr;
#[path = "crash_reserve.rs"]
mod __slate_unit_crash_reserve;
#[path = "kexec_core.rs"]
mod __slate_unit_kexec_core;
#[path = "crash_core.rs"]
mod __slate_unit_crash_core;
#[path = "kexec.rs"]
mod __slate_unit_kexec;
#[path = "compat.rs"]
mod __slate_unit_compat;
#[path = "utsname.rs"]
mod __slate_unit_utsname;
#[path = "pid_namespace.rs"]
mod __slate_unit_pid_namespace;
#[path = "stop_machine.rs"]
mod __slate_unit_stop_machine;
#[path = "audit.rs"]
mod __slate_unit_audit;
#[path = "auditfilter.rs"]
mod __slate_unit_auditfilter;
#[path = "auditsc.rs"]
mod __slate_unit_auditsc;
#[path = "audit_watch.rs"]
mod __slate_unit_audit_watch;
#[path = "audit_fsnotify.rs"]
mod __slate_unit_audit_fsnotify;
#[path = "audit_tree.rs"]
mod __slate_unit_audit_tree;
#[path = "kprobes.rs"]
mod __slate_unit_kprobes;
#[path = "seccomp.rs"]
mod __slate_unit_seccomp;
#[path = "relay.rs"]
mod __slate_unit_relay;
#[path = "utsname_sysctl.rs"]
mod __slate_unit_utsname_sysctl;
#[path = "delayacct.rs"]
mod __slate_unit_delayacct;
#[path = "taskstats.rs"]
mod __slate_unit_taskstats;
#[path = "tsacct.rs"]
mod __slate_unit_tsacct;
#[path = "tracepoint.rs"]
mod __slate_unit_tracepoint;
#[path = "irq_work.rs"]
mod __slate_unit_irq_work;
#[path = "static_call.rs"]
mod __slate_unit_static_call;
#[path = "static_call_inline.rs"]
mod __slate_unit_static_call_inline;
#[path = "padata.rs"]
mod __slate_unit_padata;
#[path = "jump_label.rs"]
mod __slate_unit_jump_label;
#[path = "context_tracking.rs"]
mod __slate_unit_context_tracking;
#[path = "iomem.rs"]
mod __slate_unit_iomem;
#[path = "rseq.rs"]
mod __slate_unit_rseq;

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

