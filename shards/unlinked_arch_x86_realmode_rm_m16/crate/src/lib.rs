#![no_std]
#![no_builtins]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![feature(linkage)]
#![feature(used_with_arg)]

#[path = "regs.rs"]
mod __slate_unit_regs;
#[path = "video_bios.rs"]
mod __slate_unit_video_bios;
#[path = "video_mode.rs"]
mod __slate_unit_video_mode;
#[path = "video_vesa.rs"]
mod __slate_unit_video_vesa;
#[path = "video_vga.rs"]
mod __slate_unit_video_vga;
#[path = "wakemain.rs"]
mod __slate_unit_wakemain;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

