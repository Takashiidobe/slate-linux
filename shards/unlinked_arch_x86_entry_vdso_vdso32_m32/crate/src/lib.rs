#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(named_asm_labels)]
#![feature(linkage)]

#[path = "vclock_gettime.rs"]
mod __slate_unit_vclock_gettime;
#[path = "vfutex.rs"]
mod __slate_unit_vfutex;
#[path = "vgetcpu.rs"]
mod __slate_unit_vgetcpu;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

