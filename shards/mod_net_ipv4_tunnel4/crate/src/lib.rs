#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![feature(linkage)]
#![feature(used_with_arg)]

#[path = "tunnel4_mod.rs"]
mod __slate_unit_tunnel4_mod;
#[path = "tunnel4.rs"]
mod __slate_unit_tunnel4;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

