#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "__slate_unit_lib.rs"]
mod __slate_unit_lib;
#[path = "class.rs"]
mod __slate_unit_class;
#[path = "interface.rs"]
mod __slate_unit_interface;
#[path = "nvmem.rs"]
mod __slate_unit_nvmem;
#[path = "dev.rs"]
mod __slate_unit_dev;
#[path = "proc.rs"]
mod __slate_unit_proc;
#[path = "sysfs.rs"]
mod __slate_unit_sysfs;
#[path = "rtc_mc146818_lib.rs"]
mod __slate_unit_rtc_mc146818_lib;
#[path = "rtc_cmos.rs"]
mod __slate_unit_rtc_cmos;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

