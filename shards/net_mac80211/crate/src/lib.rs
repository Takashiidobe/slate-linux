#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "__slate_unit_main.rs"]
mod __slate_unit_main;
#[path = "status.rs"]
mod __slate_unit_status;
#[path = "driver_ops.rs"]
mod __slate_unit_driver_ops;
#[path = "sta_info.rs"]
mod __slate_unit_sta_info;
#[path = "wep.rs"]
mod __slate_unit_wep;
#[path = "aead_api.rs"]
mod __slate_unit_aead_api;
#[path = "wpa.rs"]
mod __slate_unit_wpa;
#[path = "scan.rs"]
mod __slate_unit_scan;
#[path = "offchannel.rs"]
mod __slate_unit_offchannel;
#[path = "ht.rs"]
mod __slate_unit_ht;
#[path = "agg_tx.rs"]
mod __slate_unit_agg_tx;
#[path = "agg_rx.rs"]
mod __slate_unit_agg_rx;
#[path = "vht.rs"]
mod __slate_unit_vht;
#[path = "he.rs"]
mod __slate_unit_he;
#[path = "s1g.rs"]
mod __slate_unit_s1g;
#[path = "ibss.rs"]
mod __slate_unit_ibss;
#[path = "iface.rs"]
mod __slate_unit_iface;
#[path = "link.rs"]
mod __slate_unit_link;
#[path = "rate.rs"]
mod __slate_unit_rate;
#[path = "tkip.rs"]
mod __slate_unit_tkip;
#[path = "aes_cmac.rs"]
mod __slate_unit_aes_cmac;
#[path = "aes_gmac.rs"]
mod __slate_unit_aes_gmac;
#[path = "fils_aead.rs"]
mod __slate_unit_fils_aead;
#[path = "cfg.rs"]
mod __slate_unit_cfg;
#[path = "ethtool.rs"]
mod __slate_unit_ethtool;
#[path = "rx.rs"]
mod __slate_unit_rx;
#[path = "spectmgmt.rs"]
mod __slate_unit_spectmgmt;
#[path = "tx.rs"]
mod __slate_unit_tx;
#[path = "key.rs"]
mod __slate_unit_key;
#[path = "util.rs"]
mod __slate_unit_util;
#[path = "parse.rs"]
mod __slate_unit_parse;
#[path = "wme.rs"]
mod __slate_unit_wme;
#[path = "chan.rs"]
mod __slate_unit_chan;
#[path = "ap.rs"]
mod __slate_unit_ap;
#[path = "trace.rs"]
mod __slate_unit_trace;
#[path = "mlme.rs"]
mod __slate_unit_mlme;
#[path = "tdls.rs"]
mod __slate_unit_tdls;
#[path = "ocb.rs"]
mod __slate_unit_ocb;
#[path = "airtime.rs"]
mod __slate_unit_airtime;
#[path = "eht.rs"]
mod __slate_unit_eht;
#[path = "uhr.rs"]
mod __slate_unit_uhr;
#[path = "nan.rs"]
mod __slate_unit_nan;
#[path = "led.rs"]
mod __slate_unit_led;
#[path = "pm.rs"]
mod __slate_unit_pm;
#[path = "rc80211_minstrel_ht.rs"]
mod __slate_unit_rc80211_minstrel_ht;
#[path = "wbrf.rs"]
mod __slate_unit_wbrf;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

