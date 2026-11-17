#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "aperture.rs"]
mod __slate_unit_aperture;
#[path = "screen_info_generic.rs"]
mod __slate_unit_screen_info_generic;
#[path = "screen_info_pci.rs"]
mod __slate_unit_screen_info_pci;
#[path = "cmdline.rs"]
mod __slate_unit_cmdline;
#[path = "nomodeset.rs"]
mod __slate_unit_nomodeset;
#[path = "hdmi.rs"]
mod __slate_unit_hdmi;
#[path = "dummycon.rs"]
mod __slate_unit_dummycon;
#[path = "vgacon.rs"]
mod __slate_unit_vgacon;
#[path = "backlight.rs"]
mod __slate_unit_backlight;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

