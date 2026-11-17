#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "virtio.rs"]
mod __slate_unit_virtio;
#[path = "virtio_ring.rs"]
mod __slate_unit_virtio_ring;
#[path = "virtio_anchor.rs"]
mod __slate_unit_virtio_anchor;
#[path = "virtio_pci_modern_dev.rs"]
mod __slate_unit_virtio_pci_modern_dev;
#[path = "virtio_pci_legacy_dev.rs"]
mod __slate_unit_virtio_pci_legacy_dev;
#[path = "virtio_pci_modern.rs"]
mod __slate_unit_virtio_pci_modern;
#[path = "virtio_pci_common.rs"]
mod __slate_unit_virtio_pci_common;
#[path = "virtio_pci_legacy.rs"]
mod __slate_unit_virtio_pci_legacy;
#[path = "virtio_pci_admin_legacy_io.rs"]
mod __slate_unit_virtio_pci_admin_legacy_io;
#[path = "virtio_input.rs"]
mod __slate_unit_virtio_input;
#[path = "virtio_dma_buf.rs"]
mod __slate_unit_virtio_dma_buf;

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

