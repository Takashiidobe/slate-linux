#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "__slate_unit_main.rs"]
mod __slate_unit_main;
#[path = "version.rs"]
mod __slate_unit_version;
#[path = "do_mounts.rs"]
mod __slate_unit_do_mounts;
#[path = "do_mounts_initrd.rs"]
mod __slate_unit_do_mounts_initrd;
#[path = "initramfs.rs"]
mod __slate_unit_initramfs;
#[path = "calibrate.rs"]
mod __slate_unit_calibrate;
#[path = "init_task.rs"]
mod __slate_unit_init_task;

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

