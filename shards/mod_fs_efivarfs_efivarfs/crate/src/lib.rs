#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "efivarfs_mod.rs"]
mod __slate_unit_efivarfs_mod;
#[path = "inode.rs"]
mod __slate_unit_inode;
#[path = "file.rs"]
mod __slate_unit_file;
#[path = "super.rs"]
mod __slate_unit_super;
#[path = "vars.rs"]
mod __slate_unit_vars;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

