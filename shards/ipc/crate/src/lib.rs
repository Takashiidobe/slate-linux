#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "compat.rs"]
mod __slate_unit_compat;
#[path = "util.rs"]
mod __slate_unit_util;
#[path = "msgutil.rs"]
mod __slate_unit_msgutil;
#[path = "msg.rs"]
mod __slate_unit_msg;
#[path = "sem.rs"]
mod __slate_unit_sem;
#[path = "shm.rs"]
mod __slate_unit_shm;
#[path = "syscall.rs"]
mod __slate_unit_syscall;
#[path = "ipc_sysctl.rs"]
mod __slate_unit_ipc_sysctl;
#[path = "mqueue.rs"]
mod __slate_unit_mqueue;
#[path = "namespace.rs"]
mod __slate_unit_namespace;
#[path = "mq_sysctl.rs"]
mod __slate_unit_mq_sysctl;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

