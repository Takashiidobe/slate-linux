#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "hsw_ips.rs"]
mod __slate_unit_hsw_ips;
#[path = "i9xx_display_sr.rs"]
mod __slate_unit_i9xx_display_sr;
#[path = "i9xx_plane.rs"]
mod __slate_unit_i9xx_plane;
#[path = "i9xx_wm.rs"]
mod __slate_unit_i9xx_wm;
#[path = "intel_alpm.rs"]
mod __slate_unit_intel_alpm;
#[path = "intel_atomic.rs"]
mod __slate_unit_intel_atomic;
#[path = "intel_audio.rs"]
mod __slate_unit_intel_audio;
#[path = "intel_bios.rs"]
mod __slate_unit_intel_bios;
#[path = "intel_bo.rs"]
mod __slate_unit_intel_bo;
#[path = "intel_bw.rs"]
mod __slate_unit_intel_bw;
#[path = "intel_casf.rs"]
mod __slate_unit_intel_casf;
#[path = "intel_cdclk.rs"]
mod __slate_unit_intel_cdclk;
#[path = "intel_cmtg.rs"]
mod __slate_unit_intel_cmtg;
#[path = "intel_color.rs"]
mod __slate_unit_intel_color;
#[path = "intel_colorop.rs"]
mod __slate_unit_intel_colorop;
#[path = "intel_color_pipeline.rs"]
mod __slate_unit_intel_color_pipeline;
#[path = "intel_combo_phy.rs"]
mod __slate_unit_intel_combo_phy;
#[path = "intel_connector.rs"]
mod __slate_unit_intel_connector;
#[path = "intel_crtc.rs"]
mod __slate_unit_intel_crtc;
#[path = "intel_crtc_state_dump.rs"]
mod __slate_unit_intel_crtc_state_dump;
#[path = "intel_cursor.rs"]
mod __slate_unit_intel_cursor;
#[path = "intel_dbuf_bw.rs"]
mod __slate_unit_intel_dbuf_bw;
#[path = "intel_de.rs"]
mod __slate_unit_intel_de;
#[path = "intel_display_clock_gating.rs"]
mod __slate_unit_intel_display_clock_gating;
#[path = "intel_display.rs"]
mod __slate_unit_intel_display;
#[path = "intel_display_conversion.rs"]
mod __slate_unit_intel_display_conversion;
#[path = "intel_display_driver.rs"]
mod __slate_unit_intel_display_driver;
#[path = "intel_display_irq.rs"]
mod __slate_unit_intel_display_irq;
#[path = "intel_display_params.rs"]
mod __slate_unit_intel_display_params;
#[path = "intel_display_power.rs"]
mod __slate_unit_intel_display_power;
#[path = "intel_display_power_map.rs"]
mod __slate_unit_intel_display_power_map;
#[path = "intel_display_power_well.rs"]
mod __slate_unit_intel_display_power_well;
#[path = "intel_display_reset.rs"]
mod __slate_unit_intel_display_reset;
#[path = "intel_display_rpm.rs"]
mod __slate_unit_intel_display_rpm;
#[path = "intel_display_rps.rs"]
mod __slate_unit_intel_display_rps;
#[path = "intel_display_snapshot.rs"]
mod __slate_unit_intel_display_snapshot;
#[path = "intel_display_utils.rs"]
mod __slate_unit_intel_display_utils;
#[path = "intel_display_wa.rs"]
mod __slate_unit_intel_display_wa;
#[path = "intel_dmc.rs"]
mod __slate_unit_intel_dmc;
#[path = "intel_dmc_wl.rs"]
mod __slate_unit_intel_dmc_wl;
#[path = "intel_dpio_phy.rs"]
mod __slate_unit_intel_dpio_phy;
#[path = "intel_dpll.rs"]
mod __slate_unit_intel_dpll;
#[path = "intel_dpll_mgr.rs"]
mod __slate_unit_intel_dpll_mgr;
#[path = "intel_dpt.rs"]
mod __slate_unit_intel_dpt;
#[path = "intel_dram.rs"]
mod __slate_unit_intel_dram;
#[path = "intel_drrs.rs"]
mod __slate_unit_intel_drrs;
#[path = "intel_dsb.rs"]
mod __slate_unit_intel_dsb;
#[path = "intel_fb.rs"]
mod __slate_unit_intel_fb;
#[path = "intel_fbc.rs"]
mod __slate_unit_intel_fbc;
#[path = "intel_fdi.rs"]
mod __slate_unit_intel_fdi;
#[path = "intel_fifo_underrun.rs"]
mod __slate_unit_intel_fifo_underrun;
#[path = "intel_flipq.rs"]
mod __slate_unit_intel_flipq;
#[path = "intel_frontbuffer.rs"]
mod __slate_unit_intel_frontbuffer;
#[path = "intel_global_state.rs"]
mod __slate_unit_intel_global_state;
#[path = "intel_hdcp.rs"]
mod __slate_unit_intel_hdcp;
#[path = "intel_hdcp_gsc_message.rs"]
mod __slate_unit_intel_hdcp_gsc_message;
#[path = "intel_hotplug.rs"]
mod __slate_unit_intel_hotplug;
#[path = "intel_hotplug_irq.rs"]
mod __slate_unit_intel_hotplug_irq;
#[path = "intel_hti.rs"]
mod __slate_unit_intel_hti;
#[path = "intel_initial_plane.rs"]
mod __slate_unit_intel_initial_plane;
#[path = "intel_link_bw.rs"]
mod __slate_unit_intel_link_bw;
#[path = "intel_load_detect.rs"]
mod __slate_unit_intel_load_detect;
#[path = "intel_lpe_audio.rs"]
mod __slate_unit_intel_lpe_audio;
#[path = "intel_mchbar.rs"]
mod __slate_unit_intel_mchbar;
#[path = "intel_modeset_lock.rs"]
mod __slate_unit_intel_modeset_lock;
#[path = "intel_modeset_setup.rs"]
mod __slate_unit_intel_modeset_setup;
#[path = "intel_modeset_verify.rs"]
mod __slate_unit_intel_modeset_verify;
#[path = "intel_overlay.rs"]
mod __slate_unit_intel_overlay;
#[path = "intel_parent.rs"]
mod __slate_unit_intel_parent;
#[path = "intel_pch.rs"]
mod __slate_unit_intel_pch;
#[path = "intel_pch_display.rs"]
mod __slate_unit_intel_pch_display;
#[path = "intel_pch_refclk.rs"]
mod __slate_unit_intel_pch_refclk;
#[path = "intel_plane.rs"]
mod __slate_unit_intel_plane;
#[path = "intel_pmdemand.rs"]
mod __slate_unit_intel_pmdemand;
#[path = "intel_psr.rs"]
mod __slate_unit_intel_psr;
#[path = "intel_quirks.rs"]
mod __slate_unit_intel_quirks;
#[path = "intel_acpi.rs"]
mod __slate_unit_intel_acpi;
#[path = "intel_opregion.rs"]
mod __slate_unit_intel_opregion;
#[path = "intel_display_debugfs.rs"]
mod __slate_unit_intel_display_debugfs;
#[path = "intel_display_debugfs_params.rs"]
mod __slate_unit_intel_display_debugfs_params;
#[path = "intel_pipe_crc.rs"]
mod __slate_unit_intel_pipe_crc;
#[path = "dvo_ch7017.rs"]
mod __slate_unit_dvo_ch7017;
#[path = "dvo_ch7xxx.rs"]
mod __slate_unit_dvo_ch7xxx;
#[path = "dvo_ivch.rs"]
mod __slate_unit_dvo_ivch;
#[path = "dvo_ns2501.rs"]
mod __slate_unit_dvo_ns2501;
#[path = "dvo_sil164.rs"]
mod __slate_unit_dvo_sil164;
#[path = "dvo_tfp410.rs"]
mod __slate_unit_dvo_tfp410;
#[path = "g4x_dp.rs"]
mod __slate_unit_g4x_dp;
#[path = "g4x_hdmi.rs"]
mod __slate_unit_g4x_hdmi;
#[path = "icl_dsi.rs"]
mod __slate_unit_icl_dsi;
#[path = "intel_backlight.rs"]
mod __slate_unit_intel_backlight;
#[path = "intel_crt.rs"]
mod __slate_unit_intel_crt;
#[path = "intel_cx0_phy.rs"]
mod __slate_unit_intel_cx0_phy;
#[path = "intel_ddi.rs"]
mod __slate_unit_intel_ddi;
#[path = "intel_ddi_buf_trans.rs"]
mod __slate_unit_intel_ddi_buf_trans;
#[path = "intel_display_device.rs"]
mod __slate_unit_intel_display_device;
#[path = "intel_display_trace.rs"]
mod __slate_unit_intel_display_trace;
#[path = "intel_dkl_phy.rs"]
mod __slate_unit_intel_dkl_phy;
#[path = "intel_dp.rs"]
mod __slate_unit_intel_dp;
#[path = "intel_dp_aux.rs"]
mod __slate_unit_intel_dp_aux;
#[path = "intel_dp_aux_backlight.rs"]
mod __slate_unit_intel_dp_aux_backlight;
#[path = "intel_dp_hdcp.rs"]
mod __slate_unit_intel_dp_hdcp;
#[path = "intel_dp_link_caps.rs"]
mod __slate_unit_intel_dp_link_caps;
#[path = "intel_dp_link_training.rs"]
mod __slate_unit_intel_dp_link_training;
#[path = "intel_dp_mst.rs"]
mod __slate_unit_intel_dp_mst;
#[path = "intel_dp_test.rs"]
mod __slate_unit_intel_dp_test;
#[path = "intel_dsi.rs"]
mod __slate_unit_intel_dsi;
#[path = "intel_dsi_dcs_backlight.rs"]
mod __slate_unit_intel_dsi_dcs_backlight;
#[path = "intel_dsi_vbt.rs"]
mod __slate_unit_intel_dsi_vbt;
#[path = "intel_dvo.rs"]
mod __slate_unit_intel_dvo;
#[path = "intel_encoder.rs"]
mod __slate_unit_intel_encoder;
#[path = "intel_gmbus.rs"]
mod __slate_unit_intel_gmbus;
#[path = "intel_hdmi.rs"]
mod __slate_unit_intel_hdmi;
#[path = "intel_lspcon.rs"]
mod __slate_unit_intel_lspcon;
#[path = "intel_lt_phy.rs"]
mod __slate_unit_intel_lt_phy;
#[path = "intel_lvds.rs"]
mod __slate_unit_intel_lvds;
#[path = "intel_panel.rs"]
mod __slate_unit_intel_panel;
#[path = "intel_pfit.rs"]
mod __slate_unit_intel_pfit;
#[path = "intel_pps.rs"]
mod __slate_unit_intel_pps;
#[path = "intel_qp_tables.rs"]
mod __slate_unit_intel_qp_tables;

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

