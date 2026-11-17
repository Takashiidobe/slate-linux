#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "common.rs"]
mod __slate_unit_common;
#[path = "debug.rs"]
mod __slate_unit_debug;
#[path = "core_usb.rs"]
mod __slate_unit_core_usb;
#[path = "hub.rs"]
mod __slate_unit_hub;
#[path = "hcd.rs"]
mod __slate_unit_hcd;
#[path = "urb.rs"]
mod __slate_unit_urb;
#[path = "message.rs"]
mod __slate_unit_message;
#[path = "driver.rs"]
mod __slate_unit_driver;
#[path = "config.rs"]
mod __slate_unit_config;
#[path = "file.rs"]
mod __slate_unit_file;
#[path = "buffer.rs"]
mod __slate_unit_buffer;
#[path = "sysfs.rs"]
mod __slate_unit_sysfs;
#[path = "endpoint.rs"]
mod __slate_unit_endpoint;
#[path = "devio.rs"]
mod __slate_unit_devio;
#[path = "notify.rs"]
mod __slate_unit_notify;
#[path = "generic.rs"]
mod __slate_unit_generic;
#[path = "quirks.rs"]
mod __slate_unit_quirks;
#[path = "devices.rs"]
mod __slate_unit_devices;
#[path = "phy.rs"]
mod __slate_unit_phy;
#[path = "port.rs"]
mod __slate_unit_port;
#[path = "trace.rs"]
mod __slate_unit_trace;
#[path = "hcd_pci.rs"]
mod __slate_unit_hcd_pci;
#[path = "usb_acpi.rs"]
mod __slate_unit_usb_acpi;
#[path = "mon_main.rs"]
mod __slate_unit_mon_main;
#[path = "mon_stat.rs"]
mod __slate_unit_mon_stat;
#[path = "mon_text.rs"]
mod __slate_unit_mon_text;
#[path = "mon_bin.rs"]
mod __slate_unit_mon_bin;
#[path = "pci_quirks.rs"]
mod __slate_unit_pci_quirks;
#[path = "ehci_hcd.rs"]
mod __slate_unit_ehci_hcd;
#[path = "ehci_pci.rs"]
mod __slate_unit_ehci_pci;
#[path = "ohci_hcd.rs"]
mod __slate_unit_ohci_hcd;
#[path = "ohci_pci.rs"]
mod __slate_unit_ohci_pci;
#[path = "uhci_hcd.rs"]
mod __slate_unit_uhci_hcd;
#[path = "xhci.rs"]
mod __slate_unit_xhci;
#[path = "xhci_mem.rs"]
mod __slate_unit_xhci_mem;
#[path = "xhci_ext_caps.rs"]
mod __slate_unit_xhci_ext_caps;
#[path = "xhci_ring.rs"]
mod __slate_unit_xhci_ring;
#[path = "xhci_hub.rs"]
mod __slate_unit_xhci_hub;
#[path = "xhci_dbg.rs"]
mod __slate_unit_xhci_dbg;
#[path = "xhci_trace.rs"]
mod __slate_unit_xhci_trace;
#[path = "xhci_debugfs.rs"]
mod __slate_unit_xhci_debugfs;
#[path = "xhci_pci.rs"]
mod __slate_unit_xhci_pci;
#[path = "usblp.rs"]
mod __slate_unit_usblp;
#[path = "scsiglue.rs"]
mod __slate_unit_scsiglue;
#[path = "protocol.rs"]
mod __slate_unit_protocol;
#[path = "transport.rs"]
mod __slate_unit_transport;
#[path = "storage_usb.rs"]
mod __slate_unit_storage_usb;
#[path = "initializers.rs"]
mod __slate_unit_initializers;
#[path = "sierra_ms.rs"]
mod __slate_unit_sierra_ms;
#[path = "option_ms.rs"]
mod __slate_unit_option_ms;
#[path = "usual_tables.rs"]
mod __slate_unit_usual_tables;
#[path = "ehci_dbgp.rs"]
mod __slate_unit_ehci_dbgp;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

