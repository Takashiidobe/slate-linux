#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "drm_atomic.rs"]
mod __slate_unit_drm_atomic;
#[path = "drm_atomic_uapi.rs"]
mod __slate_unit_drm_atomic_uapi;
#[path = "drm_auth.rs"]
mod __slate_unit_drm_auth;
#[path = "drm_blend.rs"]
mod __slate_unit_drm_blend;
#[path = "drm_bridge.rs"]
mod __slate_unit_drm_bridge;
#[path = "drm_cache.rs"]
mod __slate_unit_drm_cache;
#[path = "drm_color_mgmt.rs"]
mod __slate_unit_drm_color_mgmt;
#[path = "drm_colorop.rs"]
mod __slate_unit_drm_colorop;
#[path = "drm_connector.rs"]
mod __slate_unit_drm_connector;
#[path = "drm_crtc.rs"]
mod __slate_unit_drm_crtc;
#[path = "drm_displayid.rs"]
mod __slate_unit_drm_displayid;
#[path = "drm_drv.rs"]
mod __slate_unit_drm_drv;
#[path = "drm_dumb_buffers.rs"]
mod __slate_unit_drm_dumb_buffers;
#[path = "drm_edid.rs"]
mod __slate_unit_drm_edid;
#[path = "drm_eld.rs"]
mod __slate_unit_drm_eld;
#[path = "drm_encoder.rs"]
mod __slate_unit_drm_encoder;
#[path = "drm_file.rs"]
mod __slate_unit_drm_file;
#[path = "drm_fourcc.rs"]
mod __slate_unit_drm_fourcc;
#[path = "drm_framebuffer.rs"]
mod __slate_unit_drm_framebuffer;
#[path = "drm_gem.rs"]
mod __slate_unit_drm_gem;
#[path = "drm_ioctl.rs"]
mod __slate_unit_drm_ioctl;
#[path = "drm_lease.rs"]
mod __slate_unit_drm_lease;
#[path = "drm_managed.rs"]
mod __slate_unit_drm_managed;
#[path = "drm_mm.rs"]
mod __slate_unit_drm_mm;
#[path = "drm_mode_config.rs"]
mod __slate_unit_drm_mode_config;
#[path = "drm_mode_object.rs"]
mod __slate_unit_drm_mode_object;
#[path = "drm_modes.rs"]
mod __slate_unit_drm_modes;
#[path = "drm_modeset_lock.rs"]
mod __slate_unit_drm_modeset_lock;
#[path = "drm_plane.rs"]
mod __slate_unit_drm_plane;
#[path = "drm_prime.rs"]
mod __slate_unit_drm_prime;
#[path = "drm_print.rs"]
mod __slate_unit_drm_print;
#[path = "drm_property.rs"]
mod __slate_unit_drm_property;
#[path = "drm_rect.rs"]
mod __slate_unit_drm_rect;
#[path = "drm_syncobj.rs"]
mod __slate_unit_drm_syncobj;
#[path = "drm_sysfs.rs"]
mod __slate_unit_drm_sysfs;
#[path = "drm_trace_points.rs"]
mod __slate_unit_drm_trace_points;
#[path = "drm_vblank.rs"]
mod __slate_unit_drm_vblank;
#[path = "drm_vblank_work.rs"]
mod __slate_unit_drm_vblank_work;
#[path = "drm_vma_manager.rs"]
mod __slate_unit_drm_vma_manager;
#[path = "drm_writeback.rs"]
mod __slate_unit_drm_writeback;
#[path = "drm_ioc32.rs"]
mod __slate_unit_drm_ioc32;
#[path = "drm_panel.rs"]
mod __slate_unit_drm_panel;
#[path = "drm_pci.rs"]
mod __slate_unit_drm_pci;
#[path = "drm_debugfs.rs"]
mod __slate_unit_drm_debugfs;
#[path = "drm_debugfs_crc.rs"]
mod __slate_unit_drm_debugfs_crc;
#[path = "drm_panel_orientation_quirks.rs"]
mod __slate_unit_drm_panel_orientation_quirks;
#[path = "drm_buddy.rs"]
mod __slate_unit_drm_buddy;
#[path = "drm_gem_shmem_helper.rs"]
mod __slate_unit_drm_gem_shmem_helper;
#[path = "drm_atomic_helper.rs"]
mod __slate_unit_drm_atomic_helper;
#[path = "drm_atomic_state_helper.rs"]
mod __slate_unit_drm_atomic_state_helper;
#[path = "drm_bridge_helper.rs"]
mod __slate_unit_drm_bridge_helper;
#[path = "drm_crtc_helper.rs"]
mod __slate_unit_drm_crtc_helper;
#[path = "drm_damage_helper.rs"]
mod __slate_unit_drm_damage_helper;
#[path = "drm_flip_work.rs"]
mod __slate_unit_drm_flip_work;
#[path = "drm_format_helper.rs"]
mod __slate_unit_drm_format_helper;
#[path = "drm_gem_atomic_helper.rs"]
mod __slate_unit_drm_gem_atomic_helper;
#[path = "drm_gem_framebuffer_helper.rs"]
mod __slate_unit_drm_gem_framebuffer_helper;
#[path = "drm_kms_helper_common.rs"]
mod __slate_unit_drm_kms_helper_common;
#[path = "drm_modeset_helper.rs"]
mod __slate_unit_drm_modeset_helper;
#[path = "drm_plane_helper.rs"]
mod __slate_unit_drm_plane_helper;
#[path = "drm_probe_helper.rs"]
mod __slate_unit_drm_probe_helper;
#[path = "drm_self_refresh_helper.rs"]
mod __slate_unit_drm_self_refresh_helper;
#[path = "drm_simple_kms_helper.rs"]
mod __slate_unit_drm_simple_kms_helper;
#[path = "drm_vblank_helper.rs"]
mod __slate_unit_drm_vblank_helper;
#[path = "drm_mipi_dsi.rs"]
mod __slate_unit_drm_mipi_dsi;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

