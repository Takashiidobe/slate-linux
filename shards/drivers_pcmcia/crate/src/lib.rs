#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "cs.rs"]
mod __slate_unit_cs;
#[path = "socket_sysfs.rs"]
mod __slate_unit_socket_sysfs;
#[path = "cardbus.rs"]
mod __slate_unit_cardbus;
#[path = "ds.rs"]
mod __slate_unit_ds;
#[path = "pcmcia_resource.rs"]
mod __slate_unit_pcmcia_resource;
#[path = "cistpl.rs"]
mod __slate_unit_cistpl;
#[path = "pcmcia_cis.rs"]
mod __slate_unit_pcmcia_cis;
#[path = "rsrc_mgr.rs"]
mod __slate_unit_rsrc_mgr;
#[path = "rsrc_nonstatic.rs"]
mod __slate_unit_rsrc_nonstatic;
#[path = "yenta_socket.rs"]
mod __slate_unit_yenta_socket;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

