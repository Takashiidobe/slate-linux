#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "i915_config.rs"]
mod __slate_unit_i915_config;
#[path = "i915_driver.rs"]
mod __slate_unit_i915_driver;
#[path = "i915_drm_client.rs"]
mod __slate_unit_i915_drm_client;
#[path = "i915_edram.rs"]
mod __slate_unit_i915_edram;
#[path = "i915_freq.rs"]
mod __slate_unit_i915_freq;
#[path = "i915_getparam.rs"]
mod __slate_unit_i915_getparam;
#[path = "i915_gmch.rs"]
mod __slate_unit_i915_gmch;
#[path = "i915_ioctl.rs"]
mod __slate_unit_i915_ioctl;
#[path = "i915_irq.rs"]
mod __slate_unit_i915_irq;
#[path = "i915_mitigations.rs"]
mod __slate_unit_i915_mitigations;
#[path = "i915_mmio_range.rs"]
mod __slate_unit_i915_mmio_range;
#[path = "i915_module.rs"]
mod __slate_unit_i915_module;
#[path = "i915_params.rs"]
mod __slate_unit_i915_params;
#[path = "i915_pci.rs"]
mod __slate_unit_i915_pci;
#[path = "i915_scatterlist.rs"]
mod __slate_unit_i915_scatterlist;
#[path = "i915_switcheroo.rs"]
mod __slate_unit_i915_switcheroo;
#[path = "i915_sysfs.rs"]
mod __slate_unit_i915_sysfs;
#[path = "i915_timer_util.rs"]
mod __slate_unit_i915_timer_util;
#[path = "i915_utils.rs"]
mod __slate_unit_i915_utils;
#[path = "intel_clock_gating.rs"]
mod __slate_unit_intel_clock_gating;
#[path = "intel_cpu_info.rs"]
mod __slate_unit_intel_cpu_info;
#[path = "intel_device_info.rs"]
mod __slate_unit_intel_device_info;
#[path = "intel_memory_region.rs"]
mod __slate_unit_intel_memory_region;
#[path = "intel_pcode.rs"]
mod __slate_unit_intel_pcode;
#[path = "intel_region_ttm.rs"]
mod __slate_unit_intel_region_ttm;
#[path = "intel_runtime_pm.rs"]
mod __slate_unit_intel_runtime_pm;
#[path = "intel_step.rs"]
mod __slate_unit_intel_step;
#[path = "intel_uncore.rs"]
mod __slate_unit_intel_uncore;
#[path = "intel_uncore_trace.rs"]
mod __slate_unit_intel_uncore_trace;
#[path = "intel_wakeref.rs"]
mod __slate_unit_intel_wakeref;
#[path = "vlv_iosf_sb.rs"]
mod __slate_unit_vlv_iosf_sb;
#[path = "vlv_suspend.rs"]
mod __slate_unit_vlv_suspend;
#[path = "i915_memcpy.rs"]
mod __slate_unit_i915_memcpy;
#[path = "i915_mm.rs"]
mod __slate_unit_i915_mm;
#[path = "i915_sw_fence.rs"]
mod __slate_unit_i915_sw_fence;
#[path = "i915_sw_fence_work.rs"]
mod __slate_unit_i915_sw_fence_work;
#[path = "i915_syncmap.rs"]
mod __slate_unit_i915_syncmap;
#[path = "i915_user_extensions.rs"]
mod __slate_unit_i915_user_extensions;
#[path = "i915_ioc32.rs"]
mod __slate_unit_i915_ioc32;
#[path = "i915_debugfs.rs"]
mod __slate_unit_i915_debugfs;
#[path = "i915_debugfs_params.rs"]
mod __slate_unit_i915_debugfs_params;
#[path = "i915_pmu.rs"]
mod __slate_unit_i915_pmu;
#[path = "i915_bo.rs"]
mod __slate_unit_i915_bo;
#[path = "i915_display_pc8.rs"]
mod __slate_unit_i915_display_pc8;
#[path = "i915_dpt.rs"]
mod __slate_unit_i915_dpt;
#[path = "i915_dsb_buffer.rs"]
mod __slate_unit_i915_dsb_buffer;
#[path = "i915_fb_pin.rs"]
mod __slate_unit_i915_fb_pin;
#[path = "i915_hdcp_gsc.rs"]
mod __slate_unit_i915_hdcp_gsc;
#[path = "i915_initial_plane.rs"]
mod __slate_unit_i915_initial_plane;
#[path = "i915_overlay.rs"]
mod __slate_unit_i915_overlay;
#[path = "i915_active.rs"]
mod __slate_unit_i915_active;
#[path = "i915_cmd_parser.rs"]
mod __slate_unit_i915_cmd_parser;
#[path = "i915_deps.rs"]
mod __slate_unit_i915_deps;
#[path = "i915_gem.rs"]
mod __slate_unit_i915_gem;
#[path = "i915_gem_evict.rs"]
mod __slate_unit_i915_gem_evict;
#[path = "i915_gem_gtt.rs"]
mod __slate_unit_i915_gem_gtt;
#[path = "i915_gem_ww.rs"]
mod __slate_unit_i915_gem_ww;
#[path = "i915_query.rs"]
mod __slate_unit_i915_query;
#[path = "i915_request.rs"]
mod __slate_unit_i915_request;
#[path = "i915_scheduler.rs"]
mod __slate_unit_i915_scheduler;
#[path = "i915_trace_points.rs"]
mod __slate_unit_i915_trace_points;
#[path = "i915_ttm_buddy_manager.rs"]
mod __slate_unit_i915_ttm_buddy_manager;
#[path = "i915_vma.rs"]
mod __slate_unit_i915_vma;
#[path = "i915_vma_resource.rs"]
mod __slate_unit_i915_vma_resource;
#[path = "i915_hwmon.rs"]
mod __slate_unit_i915_hwmon;
#[path = "i915_perf.rs"]
mod __slate_unit_i915_perf;
#[path = "i915_gpu_error.rs"]
mod __slate_unit_i915_gpu_error;
#[path = "i915_vgpu.rs"]
mod __slate_unit_i915_vgpu;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

