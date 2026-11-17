#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "task_mmu.rs"]
mod __slate_unit_task_mmu;
#[path = "inode.rs"]
mod __slate_unit_inode;
#[path = "root.rs"]
mod __slate_unit_root;
#[path = "base.rs"]
mod __slate_unit_base;
#[path = "generic.rs"]
mod __slate_unit_generic;
#[path = "array.rs"]
mod __slate_unit_array;
#[path = "fd.rs"]
mod __slate_unit_fd;
#[path = "proc_tty.rs"]
mod __slate_unit_proc_tty;
#[path = "cmdline.rs"]
mod __slate_unit_cmdline;
#[path = "consoles.rs"]
mod __slate_unit_consoles;
#[path = "cpuinfo.rs"]
mod __slate_unit_cpuinfo;
#[path = "devices.rs"]
mod __slate_unit_devices;
#[path = "loadavg.rs"]
mod __slate_unit_loadavg;
#[path = "meminfo.rs"]
mod __slate_unit_meminfo;
#[path = "stat.rs"]
mod __slate_unit_stat;
#[path = "uptime.rs"]
mod __slate_unit_uptime;
#[path = "util.rs"]
mod __slate_unit_util;
#[path = "version.rs"]
mod __slate_unit_version;
#[path = "softirqs.rs"]
mod __slate_unit_softirqs;
#[path = "namespaces.rs"]
mod __slate_unit_namespaces;
#[path = "self.rs"]
mod __slate_unit_self;
#[path = "thread_self.rs"]
mod __slate_unit_thread_self;
#[path = "proc_sysctl.rs"]
mod __slate_unit_proc_sysctl;
#[path = "proc_net.rs"]
mod __slate_unit_proc_net;
#[path = "kcore.rs"]
mod __slate_unit_kcore;
#[path = "vmcore.rs"]
mod __slate_unit_vmcore;
#[path = "kmsg.rs"]
mod __slate_unit_kmsg;
#[path = "page.rs"]
mod __slate_unit_page;

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

