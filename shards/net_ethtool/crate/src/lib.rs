#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "ioctl.rs"]
mod __slate_unit_ioctl;
#[path = "common.rs"]
mod __slate_unit_common;
#[path = "netlink.rs"]
mod __slate_unit_netlink;
#[path = "bitset.rs"]
mod __slate_unit_bitset;
#[path = "strset.rs"]
mod __slate_unit_strset;
#[path = "linkinfo.rs"]
mod __slate_unit_linkinfo;
#[path = "linkmodes.rs"]
mod __slate_unit_linkmodes;
#[path = "rss.rs"]
mod __slate_unit_rss;
#[path = "linkstate.rs"]
mod __slate_unit_linkstate;
#[path = "debug.rs"]
mod __slate_unit_debug;
#[path = "wol.rs"]
mod __slate_unit_wol;
#[path = "features.rs"]
mod __slate_unit_features;
#[path = "privflags.rs"]
mod __slate_unit_privflags;
#[path = "rings.rs"]
mod __slate_unit_rings;
#[path = "channels.rs"]
mod __slate_unit_channels;
#[path = "coalesce.rs"]
mod __slate_unit_coalesce;
#[path = "pause.rs"]
mod __slate_unit_pause;
#[path = "eee.rs"]
mod __slate_unit_eee;
#[path = "tsinfo.rs"]
mod __slate_unit_tsinfo;
#[path = "cabletest.rs"]
mod __slate_unit_cabletest;
#[path = "tunnels.rs"]
mod __slate_unit_tunnels;
#[path = "fec.rs"]
mod __slate_unit_fec;
#[path = "eeprom.rs"]
mod __slate_unit_eeprom;
#[path = "stats.rs"]
mod __slate_unit_stats;
#[path = "phc_vclocks.rs"]
mod __slate_unit_phc_vclocks;
#[path = "mm.rs"]
mod __slate_unit_mm;
#[path = "module.rs"]
mod __slate_unit_module;
#[path = "cmis_fw_update.rs"]
mod __slate_unit_cmis_fw_update;
#[path = "cmis_cdb.rs"]
mod __slate_unit_cmis_cdb;
#[path = "pse_pd.rs"]
mod __slate_unit_pse_pd;
#[path = "plca.rs"]
mod __slate_unit_plca;
#[path = "phy.rs"]
mod __slate_unit_phy;
#[path = "tsconfig.rs"]
mod __slate_unit_tsconfig;
#[path = "mse.rs"]
mod __slate_unit_mse;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

