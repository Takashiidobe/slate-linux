#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "panel.rs"]
mod __slate_unit_panel;
#[path = "drm_display_helper_mod.rs"]
mod __slate_unit_drm_display_helper_mod;
#[path = "drm_dp_dual_mode_helper.rs"]
mod __slate_unit_drm_dp_dual_mode_helper;
#[path = "drm_dp_helper.rs"]
mod __slate_unit_drm_dp_helper;
#[path = "drm_dp_mst_topology.rs"]
mod __slate_unit_drm_dp_mst_topology;
#[path = "drm_dsc_helper.rs"]
mod __slate_unit_drm_dsc_helper;
#[path = "drm_hdcp_helper.rs"]
mod __slate_unit_drm_hdcp_helper;
#[path = "drm_hdmi_helper.rs"]
mod __slate_unit_drm_hdmi_helper;
#[path = "drm_scdc_helper.rs"]
mod __slate_unit_drm_scdc_helper;
#[path = "ttm_tt.rs"]
mod __slate_unit_ttm_tt;
#[path = "ttm_bo.rs"]
mod __slate_unit_ttm_bo;
#[path = "ttm_bo_util.rs"]
mod __slate_unit_ttm_bo_util;
#[path = "ttm_bo_vm.rs"]
mod __slate_unit_ttm_bo_vm;
#[path = "ttm_module.rs"]
mod __slate_unit_ttm_module;
#[path = "ttm_execbuf_util.rs"]
mod __slate_unit_ttm_execbuf_util;
#[path = "ttm_range_manager.rs"]
mod __slate_unit_ttm_range_manager;
#[path = "ttm_resource.rs"]
mod __slate_unit_ttm_resource;
#[path = "ttm_pool.rs"]
mod __slate_unit_ttm_pool;
#[path = "ttm_device.rs"]
mod __slate_unit_ttm_device;
#[path = "ttm_sys_manager.rs"]
mod __slate_unit_ttm_sys_manager;
#[path = "ttm_backup.rs"]
mod __slate_unit_ttm_backup;
#[path = "ttm_agp_backend.rs"]
mod __slate_unit_ttm_agp_backend;
#[path = "virtgpu_drv.rs"]
mod __slate_unit_virtgpu_drv;
#[path = "virtgpu_kms.rs"]
mod __slate_unit_virtgpu_kms;
#[path = "virtgpu_gem.rs"]
mod __slate_unit_virtgpu_gem;
#[path = "virtgpu_vram.rs"]
mod __slate_unit_virtgpu_vram;
#[path = "virtgpu_display.rs"]
mod __slate_unit_virtgpu_display;
#[path = "virtgpu_vq.rs"]
mod __slate_unit_virtgpu_vq;
#[path = "virtgpu_fence.rs"]
mod __slate_unit_virtgpu_fence;
#[path = "virtgpu_object.rs"]
mod __slate_unit_virtgpu_object;
#[path = "virtgpu_debugfs.rs"]
mod __slate_unit_virtgpu_debugfs;
#[path = "virtgpu_plane.rs"]
mod __slate_unit_virtgpu_plane;
#[path = "virtgpu_ioctl.rs"]
mod __slate_unit_virtgpu_ioctl;
#[path = "virtgpu_prime.rs"]
mod __slate_unit_virtgpu_prime;
#[path = "virtgpu_trace_points.rs"]
mod __slate_unit_virtgpu_trace_points;
#[path = "virtgpu_submit.rs"]
mod __slate_unit_virtgpu_submit;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

