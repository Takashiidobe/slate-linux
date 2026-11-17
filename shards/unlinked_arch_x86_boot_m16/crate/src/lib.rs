#![no_std]
#![no_builtins]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(used_with_arg)]

#[path = "a20.rs"]
mod __slate_unit_a20;
#[path = "cmdline.rs"]
mod __slate_unit_cmdline;
#[path = "cpu.rs"]
mod __slate_unit_cpu;
#[path = "cpucheck.rs"]
mod __slate_unit_cpucheck;
#[path = "cpuflags.rs"]
mod __slate_unit_cpuflags;
#[path = "early_serial_console.rs"]
mod __slate_unit_early_serial_console;
#[path = "edd.rs"]
mod __slate_unit_edd;
#[path = "__slate_unit_main.rs"]
mod __slate_unit_main;
#[path = "memory.rs"]
mod __slate_unit_memory;
#[path = "pm.rs"]
mod __slate_unit_pm;
#[path = "printf.rs"]
mod __slate_unit_printf;
#[path = "regs.rs"]
mod __slate_unit_regs;
#[path = "string.rs"]
mod __slate_unit_string;
#[path = "tty.rs"]
mod __slate_unit_tty;
#[path = "version.rs"]
mod __slate_unit_version;
#[path = "video_bios.rs"]
mod __slate_unit_video_bios;
#[path = "video_mode.rs"]
mod __slate_unit_video_mode;
#[path = "video_vesa.rs"]
mod __slate_unit_video_vesa;
#[path = "video_vga.rs"]
mod __slate_unit_video_vga;
#[path = "video.rs"]
mod __slate_unit_video;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

