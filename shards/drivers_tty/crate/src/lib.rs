#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "tty_io.rs"]
mod __slate_unit_tty_io;
#[path = "n_tty.rs"]
mod __slate_unit_n_tty;
#[path = "tty_ioctl.rs"]
mod __slate_unit_tty_ioctl;
#[path = "tty_ldisc.rs"]
mod __slate_unit_tty_ldisc;
#[path = "tty_buffer.rs"]
mod __slate_unit_tty_buffer;
#[path = "tty_port.rs"]
mod __slate_unit_tty_port;
#[path = "tty_mutex.rs"]
mod __slate_unit_tty_mutex;
#[path = "tty_ldsem.rs"]
mod __slate_unit_tty_ldsem;
#[path = "tty_baudrate.rs"]
mod __slate_unit_tty_baudrate;
#[path = "tty_jobctrl.rs"]
mod __slate_unit_tty_jobctrl;
#[path = "n_null.rs"]
mod __slate_unit_n_null;
#[path = "pty.rs"]
mod __slate_unit_pty;
#[path = "tty_audit.rs"]
mod __slate_unit_tty_audit;
#[path = "sysrq.rs"]
mod __slate_unit_sysrq;
#[path = "vt_ioctl.rs"]
mod __slate_unit_vt_ioctl;
#[path = "vc_screen.rs"]
mod __slate_unit_vc_screen;
#[path = "selection.rs"]
mod __slate_unit_selection;
#[path = "keyboard.rs"]
mod __slate_unit_keyboard;
#[path = "vt.rs"]
mod __slate_unit_vt;
#[path = "consolemap.rs"]
mod __slate_unit_consolemap;
#[path = "ucs.rs"]
mod __slate_unit_ucs;
#[path = "hvc_console.rs"]
mod __slate_unit_hvc_console;
#[path = "serial_core.rs"]
mod __slate_unit_serial_core;
#[path = "serial_base_bus.rs"]
mod __slate_unit_serial_base_bus;
#[path = "serial_ctrl.rs"]
mod __slate_unit_serial_ctrl;
#[path = "serial_port.rs"]
mod __slate_unit_serial_port;
#[path = "earlycon.rs"]
mod __slate_unit_earlycon;
#[path = "_8250_core.rs"]
mod __slate_unit__8250_core;
#[path = "_8250_platform.rs"]
mod __slate_unit__8250_platform;
#[path = "_8250_pnp.rs"]
mod __slate_unit__8250_pnp;
#[path = "_8250_port.rs"]
mod __slate_unit__8250_port;
#[path = "_8250_dma.rs"]
mod __slate_unit__8250_dma;
#[path = "_8250_dwlib.rs"]
mod __slate_unit__8250_dwlib;
#[path = "_8250_pcilib.rs"]
mod __slate_unit__8250_pcilib;
#[path = "_8250_rsa.rs"]
mod __slate_unit__8250_rsa;
#[path = "_8250_early.rs"]
mod __slate_unit__8250_early;
#[path = "_8250_exar.rs"]
mod __slate_unit__8250_exar;
#[path = "_8250_lpss.rs"]
mod __slate_unit__8250_lpss;
#[path = "_8250_mid.rs"]
mod __slate_unit__8250_mid;
#[path = "_8250_mxpcie.rs"]
mod __slate_unit__8250_mxpcie;
#[path = "_8250_pci.rs"]
mod __slate_unit__8250_pci;
#[path = "_8250_pericom.rs"]
mod __slate_unit__8250_pericom;

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

