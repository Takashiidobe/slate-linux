#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "mii.rs"]
mod __slate_unit_mii;
#[path = "loopback.rs"]
mod __slate_unit_loopback;
#[path = "netconsole.rs"]
mod __slate_unit_netconsole;
#[path = "stubs.rs"]
mod __slate_unit_stubs;
#[path = "mdio_bus.rs"]
mod __slate_unit_mdio_bus;
#[path = "phy_phy.rs"]
mod __slate_unit_phy_phy;
#[path = "phy_c45.rs"]
mod __slate_unit_phy_c45;
#[path = "phy_core.rs"]
mod __slate_unit_phy_core;
#[path = "phy_device.rs"]
mod __slate_unit_phy_device;
#[path = "linkmode.rs"]
mod __slate_unit_linkmode;
#[path = "phy_link_topology.rs"]
mod __slate_unit_phy_link_topology;
#[path = "phy_caps.rs"]
mod __slate_unit_phy_caps;
#[path = "mdio_bus_provider.rs"]
mod __slate_unit_mdio_bus_provider;
#[path = "phy_port.rs"]
mod __slate_unit_phy_port;
#[path = "mdio_device.rs"]
mod __slate_unit_mdio_device;
#[path = "mdio_devres.rs"]
mod __slate_unit_mdio_devres;
#[path = "phy_package.rs"]
mod __slate_unit_phy_package;
#[path = "realtek_main.rs"]
mod __slate_unit_realtek_main;
#[path = "acpi_mdio.rs"]
mod __slate_unit_acpi_mdio;
#[path = "fwnode_mdio.rs"]
mod __slate_unit_fwnode_mdio;
#[path = "virtio_net.rs"]
mod __slate_unit_virtio_net;
#[path = "tg3.rs"]
mod __slate_unit_tg3;
#[path = "e100.rs"]
mod __slate_unit_e100;
#[path = "e1000_main.rs"]
mod __slate_unit_e1000_main;
#[path = "e1000_hw.rs"]
mod __slate_unit_e1000_hw;
#[path = "e1000_ethtool.rs"]
mod __slate_unit_e1000_ethtool;
#[path = "e1000_param.rs"]
mod __slate_unit_e1000_param;
#[path = "_82571.rs"]
mod __slate_unit__82571;
#[path = "ich8lan.rs"]
mod __slate_unit_ich8lan;
#[path = "_80003es2lan.rs"]
mod __slate_unit__80003es2lan;
#[path = "mac.rs"]
mod __slate_unit_mac;
#[path = "manage.rs"]
mod __slate_unit_manage;
#[path = "nvm.rs"]
mod __slate_unit_nvm;
#[path = "e1000e_phy.rs"]
mod __slate_unit_e1000e_phy;
#[path = "param.rs"]
mod __slate_unit_param;
#[path = "ethtool.rs"]
mod __slate_unit_ethtool;
#[path = "netdev.rs"]
mod __slate_unit_netdev;
#[path = "ptp.rs"]
mod __slate_unit_ptp;
#[path = "sky2.rs"]
mod __slate_unit_sky2;
#[path = "forcedeth.rs"]
mod __slate_unit_forcedeth;
#[path = "_8139too.rs"]
mod __slate_unit__8139too;
#[path = "r8169_main.rs"]
mod __slate_unit_r8169_main;
#[path = "r8169_firmware.rs"]
mod __slate_unit_r8169_firmware;
#[path = "r8169_phy_config.rs"]
mod __slate_unit_r8169_phy_config;
#[path = "net_failover.rs"]
mod __slate_unit_net_failover;

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

