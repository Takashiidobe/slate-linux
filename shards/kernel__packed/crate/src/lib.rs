#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(asm_goto_with_outputs)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "sched_core.rs"]
mod __slate_unit_sched_core;
#[path = "fair.rs"]
mod __slate_unit_fair;
#[path = "build_policy.rs"]
mod __slate_unit_build_policy;
#[path = "build_utility.rs"]
mod __slate_unit_build_utility;
#[path = "mutex.rs"]
mod __slate_unit_mutex;
#[path = "semaphore.rs"]
mod __slate_unit_semaphore;
#[path = "rwsem.rs"]
mod __slate_unit_rwsem;
#[path = "percpu_rwsem.rs"]
mod __slate_unit_percpu_rwsem;
#[path = "spinlock.rs"]
mod __slate_unit_spinlock;
#[path = "osq_lock.rs"]
mod __slate_unit_osq_lock;
#[path = "qspinlock.rs"]
mod __slate_unit_qspinlock;
#[path = "rtmutex_api.rs"]
mod __slate_unit_rtmutex_api;
#[path = "qrwlock.rs"]
mod __slate_unit_qrwlock;
#[path = "qos.rs"]
mod __slate_unit_qos;
#[path = "power_main.rs"]
mod __slate_unit_power_main;
#[path = "console.rs"]
mod __slate_unit_console;
#[path = "process.rs"]
mod __slate_unit_process;
#[path = "suspend.rs"]
mod __slate_unit_suspend;
#[path = "hibernate.rs"]
mod __slate_unit_hibernate;
#[path = "snapshot.rs"]
mod __slate_unit_snapshot;
#[path = "swap.rs"]
mod __slate_unit_swap;
#[path = "power_user.rs"]
mod __slate_unit_power_user;
#[path = "poweroff.rs"]
mod __slate_unit_poweroff;
#[path = "printk.rs"]
mod __slate_unit_printk;
#[path = "printk_safe.rs"]
mod __slate_unit_printk_safe;
#[path = "nbcon.rs"]
mod __slate_unit_nbcon;
#[path = "printk_ringbuffer.rs"]
mod __slate_unit_printk_ringbuffer;
#[path = "sysctl.rs"]
mod __slate_unit_sysctl;
#[path = "irqdesc.rs"]
mod __slate_unit_irqdesc;
#[path = "handle.rs"]
mod __slate_unit_handle;
#[path = "manage.rs"]
mod __slate_unit_manage;
#[path = "spurious.rs"]
mod __slate_unit_spurious;
#[path = "resend.rs"]
mod __slate_unit_resend;
#[path = "chip.rs"]
mod __slate_unit_chip;
#[path = "dummychip.rs"]
mod __slate_unit_dummychip;
#[path = "devres.rs"]
mod __slate_unit_devres;
#[path = "kexec.rs"]
mod __slate_unit_kexec;
#[path = "autoprobe.rs"]
mod __slate_unit_autoprobe;
#[path = "irqdomain.rs"]
mod __slate_unit_irqdomain;
#[path = "proc.rs"]
mod __slate_unit_proc;
#[path = "migration.rs"]
mod __slate_unit_migration;
#[path = "cpuhotplug.rs"]
mod __slate_unit_cpuhotplug;
#[path = "pm.rs"]
mod __slate_unit_pm;
#[path = "msi.rs"]
mod __slate_unit_msi;
#[path = "affinity.rs"]
mod __slate_unit_affinity;
#[path = "matrix.rs"]
mod __slate_unit_matrix;
#[path = "update.rs"]
mod __slate_unit_update;
#[path = "sync.rs"]
mod __slate_unit_sync;
#[path = "srcutree.rs"]
mod __slate_unit_srcutree;
#[path = "tree.rs"]
mod __slate_unit_tree;
#[path = "rcu_segcblist.rs"]
mod __slate_unit_rcu_segcblist;
#[path = "mapping.rs"]
mod __slate_unit_mapping;
#[path = "direct.rs"]
mod __slate_unit_direct;
#[path = "ops_helpers.rs"]
mod __slate_unit_ops_helpers;
#[path = "swiotlb.rs"]
mod __slate_unit_swiotlb;
#[path = "remap.rs"]
mod __slate_unit_remap;
#[path = "common.rs"]
mod __slate_unit_common;
#[path = "syscall_common.rs"]
mod __slate_unit_syscall_common;
#[path = "syscall_user_dispatch.rs"]
mod __slate_unit_syscall_user_dispatch;
#[path = "unwind_user.rs"]
mod __slate_unit_unwind_user;
#[path = "deferred.rs"]
mod __slate_unit_deferred;
#[path = "module_main.rs"]
mod __slate_unit_module_main;
#[path = "strict_rwx.rs"]
mod __slate_unit_strict_rwx;
#[path = "kmod.rs"]
mod __slate_unit_kmod;
#[path = "tree_lookup.rs"]
mod __slate_unit_tree_lookup;
#[path = "kallsyms.rs"]
mod __slate_unit_kallsyms;
#[path = "procfs.rs"]
mod __slate_unit_procfs;
#[path = "sysfs.rs"]
mod __slate_unit_sysfs;
#[path = "time.rs"]
mod __slate_unit_time;
#[path = "timer.rs"]
mod __slate_unit_timer;
#[path = "hrtimer.rs"]
mod __slate_unit_hrtimer;
#[path = "sleep_timeout.rs"]
mod __slate_unit_sleep_timeout;
#[path = "timekeeping.rs"]
mod __slate_unit_timekeeping;
#[path = "ntp.rs"]
mod __slate_unit_ntp;
#[path = "clocksource.rs"]
mod __slate_unit_clocksource;
#[path = "jiffies.rs"]
mod __slate_unit_jiffies;
#[path = "timer_list.rs"]
mod __slate_unit_timer_list;
#[path = "timeconv.rs"]
mod __slate_unit_timeconv;
#[path = "timecounter.rs"]
mod __slate_unit_timecounter;
#[path = "alarmtimer.rs"]
mod __slate_unit_alarmtimer;
#[path = "posix_timers.rs"]
mod __slate_unit_posix_timers;
#[path = "posix_cpu_timers.rs"]
mod __slate_unit_posix_cpu_timers;
#[path = "posix_clock.rs"]
mod __slate_unit_posix_clock;
#[path = "itimer.rs"]
mod __slate_unit_itimer;
#[path = "clockevents.rs"]
mod __slate_unit_clockevents;
#[path = "tick_common.rs"]
mod __slate_unit_tick_common;
#[path = "tick_broadcast.rs"]
mod __slate_unit_tick_broadcast;
#[path = "tick_broadcast_hrtimer.rs"]
mod __slate_unit_tick_broadcast_hrtimer;
#[path = "tick_oneshot.rs"]
mod __slate_unit_tick_oneshot;
#[path = "tick_sched.rs"]
mod __slate_unit_tick_sched;
#[path = "timer_migration.rs"]
mod __slate_unit_timer_migration;
#[path = "vsyscall.rs"]
mod __slate_unit_vsyscall;
#[path = "timekeeping_debug.rs"]
mod __slate_unit_timekeeping_debug;
#[path = "time_namespace.rs"]
mod __slate_unit_time_namespace;
#[path = "namespace_vdso.rs"]
mod __slate_unit_namespace_vdso;
#[path = "futex_core.rs"]
mod __slate_unit_futex_core;
#[path = "syscalls.rs"]
mod __slate_unit_syscalls;
#[path = "pi.rs"]
mod __slate_unit_pi;
#[path = "requeue.rs"]
mod __slate_unit_requeue;
#[path = "waitwake.rs"]
mod __slate_unit_waitwake;
#[path = "cgroup.rs"]
mod __slate_unit_cgroup;
#[path = "rstat.rs"]
mod __slate_unit_rstat;
#[path = "cgroup_namespace.rs"]
mod __slate_unit_cgroup_namespace;
#[path = "cgroup_v1.rs"]
mod __slate_unit_cgroup_v1;
#[path = "freezer.rs"]
mod __slate_unit_freezer;
#[path = "legacy_freezer.rs"]
mod __slate_unit_legacy_freezer;
#[path = "pids.rs"]
mod __slate_unit_pids;
#[path = "rdma.rs"]
mod __slate_unit_rdma;
#[path = "cpuset.rs"]
mod __slate_unit_cpuset;
#[path = "misc.rs"]
mod __slate_unit_misc;
#[path = "debug.rs"]
mod __slate_unit_debug;
#[path = "bpf_core.rs"]
mod __slate_unit_bpf_core;
#[path = "events_core.rs"]
mod __slate_unit_events_core;
#[path = "ring_buffer.rs"]
mod __slate_unit_ring_buffer;
#[path = "callchain.rs"]
mod __slate_unit_callchain;
#[path = "hw_breakpoint.rs"]
mod __slate_unit_hw_breakpoint;
#[path = "uprobes.rs"]
mod __slate_unit_uprobes;

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

