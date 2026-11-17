#![no_std]
#![no_builtins]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]

#[path = "acpi.rs"]
mod __slate_unit_acpi;
#[path = "cmdline.rs"]
mod __slate_unit_cmdline;
#[path = "cpuflags.rs"]
mod __slate_unit_cpuflags;
#[path = "early_serial_console.rs"]
mod __slate_unit_early_serial_console;
#[path = "efi.rs"]
mod __slate_unit_efi;
#[path = "error.rs"]
mod __slate_unit_error;
#[path = "ident_map_64.rs"]
mod __slate_unit_ident_map_64;
#[path = "idt_64.rs"]
mod __slate_unit_idt_64;
#[path = "kaslr.rs"]
mod __slate_unit_kaslr;
#[path = "misc.rs"]
mod __slate_unit_misc;
#[path = "pgtable_64.rs"]
mod __slate_unit_pgtable_64;
#[path = "string.rs"]
mod __slate_unit_string;

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

