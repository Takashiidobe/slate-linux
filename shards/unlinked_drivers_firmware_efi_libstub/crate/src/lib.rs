#![no_std]
#![no_builtins]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]

#[path = "alignedmem.rs"]
mod __slate_unit_alignedmem;
#[path = "efi_stub_helper.rs"]
mod __slate_unit_efi_stub_helper;
#[path = "file.rs"]
mod __slate_unit_file;
#[path = "gop.rs"]
mod __slate_unit_gop;
#[path = "mem.rs"]
mod __slate_unit_mem;
#[path = "pci.rs"]
mod __slate_unit_pci;
#[path = "printk.rs"]
mod __slate_unit_printk;
#[path = "random.rs"]
mod __slate_unit_random;
#[path = "randomalloc.rs"]
mod __slate_unit_randomalloc;
#[path = "secureboot.rs"]
mod __slate_unit_secureboot;
#[path = "skip_spaces.rs"]
mod __slate_unit_skip_spaces;
#[path = "smbios.rs"]
mod __slate_unit_smbios;
#[path = "tpm.rs"]
mod __slate_unit_tpm;
#[path = "vsprintf.rs"]
mod __slate_unit_vsprintf;
#[path = "x86_5lvl.rs"]
mod __slate_unit_x86_5lvl;
#[path = "x86_stub.rs"]
mod __slate_unit_x86_stub;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

