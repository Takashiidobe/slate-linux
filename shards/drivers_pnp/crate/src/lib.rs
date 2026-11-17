#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "pnp_core.rs"]
mod __slate_unit_pnp_core;
#[path = "card.rs"]
mod __slate_unit_card;
#[path = "driver.rs"]
mod __slate_unit_driver;
#[path = "resource.rs"]
mod __slate_unit_resource;
#[path = "manager.rs"]
mod __slate_unit_manager;
#[path = "support.rs"]
mod __slate_unit_support;
#[path = "interface.rs"]
mod __slate_unit_interface;
#[path = "quirks.rs"]
mod __slate_unit_quirks;
#[path = "system.rs"]
mod __slate_unit_system;
#[path = "pnpacpi_core.rs"]
mod __slate_unit_pnpacpi_core;
#[path = "rsparser.rs"]
mod __slate_unit_rsparser;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

