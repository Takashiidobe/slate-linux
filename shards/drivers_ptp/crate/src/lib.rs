#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "ptp_clock.rs"]
mod __slate_unit_ptp_clock;
#[path = "ptp_chardev.rs"]
mod __slate_unit_ptp_chardev;
#[path = "ptp_sysfs.rs"]
mod __slate_unit_ptp_sysfs;
#[path = "ptp_vclock.rs"]
mod __slate_unit_ptp_vclock;
#[path = "ptp_kvm_x86.rs"]
mod __slate_unit_ptp_kvm_x86;
#[path = "ptp_kvm_common.rs"]
mod __slate_unit_ptp_kvm_common;
#[path = "ptp_vmclock.rs"]
mod __slate_unit_ptp_vmclock;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

