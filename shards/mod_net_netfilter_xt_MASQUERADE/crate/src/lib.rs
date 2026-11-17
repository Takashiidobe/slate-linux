#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![feature(linkage)]
#![feature(used_with_arg)]

#[path = "xt_MASQUERADE_mod.rs"]
mod __slate_unit_xt_MASQUERADE_mod;
#[path = "xt_MASQUERADE.rs"]
mod __slate_unit_xt_MASQUERADE;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

